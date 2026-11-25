"""Detection API.

    VISION_MODEL=models/yolo11n.onnx uvicorn vision.api:app --port 8001

POST /detect             image file -> JSON detections + timing
POST /detect/annotated   image file -> JPEG with boxes drawn
POST /follow             image file -> velocity command to follow the best match of ?target=person
GET  /health, /metrics   liveness and latency percentiles
"""
from __future__ import annotations

import os
import threading
from collections import deque

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .detector import OnnxDetector, draw, load_image
from .tracking import IoUTracker, bearing_deg, follow_command

MAX_BYTES = 15 * 1024 * 1024


def create_app(detector=None) -> FastAPI:
    app = FastAPI(title="vision-api", version="0.1.0")
    state = {"det": detector, "lat": deque(maxlen=1000), "count": 0, "tracker": IoUTracker(min_hits=1)}
    lock = threading.Lock()

    def det():
        if state["det"] is None:
            path = os.environ.get("VISION_MODEL", "models/yolo11n.onnx")
            if not os.path.exists(path):
                raise HTTPException(503, f"model not found at {path}; see README (export to ONNX)")
            state["det"] = OnnxDetector(path)
        return state["det"]

    async def read(file: UploadFile):
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "image too large")
        try:
            return load_image(data)
        except Exception:
            raise HTTPException(400, "could not decode image")

    def record(res):
        with lock:
            state["lat"].append(res.timing.total_ms)
            state["count"] += 1

    @app.get("/health")
    def health():
        d = state["det"]
        return {"status": "ok", "model_loaded": d is not None, "provider": getattr(d, "provider", None)}

    @app.get("/metrics")
    def metrics():
        with lock:
            lat = sorted(state["lat"])
        pct = lambda p: round(lat[min(len(lat) - 1, int(p / 100 * len(lat)))], 2) if lat else None  # noqa: E731
        return {"requests": state["count"], "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99)}}

    @app.post("/detect")
    async def detect(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.01, le=1.0),
                     track: bool = False, hfov: float = 62.2):
        img = await read(file)
        res = det()(img, conf=conf)
        if track:
            with lock:
                state["tracker"].update(res.detections)
        record(res)
        out = res.to_dict()
        for d, row in zip(res.detections, out["detections"]):
            row["bearing_deg"] = round(bearing_deg(d, res.image_size[0], hfov), 1)
        return out

    @app.post("/detect/annotated")
    async def annotated(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.01, le=1.0)):
        img = await read(file)
        res = det()(img, conf=conf)
        record(res)
        return Response(draw(img, res.detections), media_type="image/jpeg",
                        headers={"X-Detections": str(len(res.detections)),
                                 "X-Latency-Ms": f"{res.timing.total_ms:.1f}"})

    @app.post("/follow")
    async def follow(file: UploadFile = File(...), target: str = "person", conf: float = 0.4):
        img = await read(file)
        res = det()(img, conf=conf)
        record(res)
        matches = [d for d in res.detections if d.label == target]
        best = max(matches, key=lambda d: d.area * d.score) if matches else None
        w, h = res.image_size
        return {"target": target, "found": best is not None, "detection": best.to_dict() if best else None,
                "command": follow_command(best, w, h)}

    return app


app = create_app()
