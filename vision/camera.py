"""Live camera loop: detect, track, and (optionally) steer the 05-quadruped towards a target.

    python -m vision.camera --model models/yolo11n.onnx                 # webcam window with boxes + FPS
    python -m vision.camera --model models/yolo11n_320.onnx --follow person --headless
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .detector import OnnxDetector
from .tracking import IoUTracker, follow_command


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/yolo11n.onnx")
    p.add_argument("--source", default="0", help="camera index or video file/URL")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--follow", help="class to follow, e.g. person; prints velocity commands")
    p.add_argument("--headless", action="store_true")
    a = p.parse_args(argv)

    import cv2  # pip install opencv-python

    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source {a.source}")
    det, tracker = OnnxDetector(a.model, conf=a.conf), IoUTracker()
    fps, last = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = frame[:, :, ::-1].copy()
        res = det(rgb)
        dets = tracker.update(res.detections)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        if a.follow:
            h, w = rgb.shape[:2]
            targets = [d for d in dets if d.label == a.follow]
            best = max(targets, key=lambda d: d.area * d.score) if targets else None
            print(follow_command(best, w, h), f"{fps:.1f} fps", flush=True)
        if not a.headless:
            for d in dets:
                cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (65, 165, 242), 2)
                tag = f"{d.label} {d.score:.2f}" + (f" #{d.track_id}" if d.track_id else "")
                cv2.putText(frame, tag, (int(d.x1), max(12, int(d.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (65, 165, 242), 1)
            cv2.putText(frame, f"{fps:.1f} FPS  {res.timing.inference_ms:.1f} ms infer", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 196, 139), 2)
            cv2.imshow("vision", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
