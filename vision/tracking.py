"""Lightweight multi-object tracker (IoU matching + constant-velocity prediction), and the geometry a
robot needs to act on a detection: which way to turn and roughly how far away it is."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .processing import Detection, iou


@dataclass
class Track:
    id: int
    box: np.ndarray       # x1 y1 x2 y2
    velocity: np.ndarray  # per-frame box delta
    class_id: int
    label: str
    hits: int = 1
    misses: int = 0

    def predict(self) -> np.ndarray:
        return self.box + self.velocity


class IoUTracker:
    """Greedy IoU association. Good enough for a robot following one or a few objects at camera
    frame rates; swap in ByteTrack/BoT-SORT when you need re-identification through occlusions."""

    def __init__(self, iou_thres: float = 0.3, max_misses: int = 10, min_hits: int = 2):
        self.iou_thres, self.max_misses, self.min_hits = iou_thres, max_misses, min_hits
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, dets: list[Detection]) -> list[Detection]:
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float64).reshape(-1, 4)
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        pairs = []
        for ti, p in enumerate(preds):
            if len(boxes):
                ious = iou(p, boxes)
                for di, v in enumerate(ious):
                    if v >= self.iou_thres and dets[di].class_id == self.tracks[ti].class_id:
                        pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            t.velocity = 0.5 * t.velocity + 0.5 * (boxes[di] - t.box)
            t.box, t.hits, t.misses = boxes[di], t.hits + 1, 0
            dets[di].track_id = t.id if t.hits >= self.min_hits else None
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
                t.box = t.predict()
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(self._next, boxes[di], np.zeros(4), d.class_id, d.label))
                if self.min_hits <= 1:
                    d.track_id = self._next
                self._next += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return dets


def bearing_deg(det: Detection, image_width: int, hfov_deg: float = 62.2) -> float:
    """Horizontal angle from the camera axis to the object centre (+ = right).
    Default HFOV is the Raspberry Pi Camera v2."""
    f = (image_width / 2) / math.tan(math.radians(hfov_deg) / 2)  # focal length in pixels
    cx = det.center[0] - image_width / 2
    return math.degrees(math.atan2(cx, f))


def distance_m(det: Detection, image_height: int, real_height_m: float, vfov_deg: float = 48.8) -> float:
    """Pinhole estimate from a known object height (e.g. a person ~1.7 m, a bottle ~0.25 m)."""
    f = (image_height / 2) / math.tan(math.radians(vfov_deg) / 2)
    h_px = max(det.y2 - det.y1, 1e-6)
    return real_height_m * f / h_px


def follow_command(det: Detection | None, image_width: int, image_height: int, target_area_frac: float = 0.12,
                   hfov_deg: float = 62.2, k_turn: float = 1.5, k_fwd: float = 400.0, max_fwd: float = 150.0) -> dict:
    """Turn a detection into a velocity command for a mobile robot (e.g. 05-quadruped):
    turn towards it, walk forward until it fills `target_area_frac` of the frame."""
    if det is None:
        return {"vx": 0.0, "yaw_rate_deg": 0.0, "status": "searching"}
    yaw = -k_turn * bearing_deg(det, image_width, hfov_deg)  # object to the right -> turn clockwise (negative)
    frac = det.area / (image_width * image_height)
    vx = max(-max_fwd / 2, min(max_fwd, k_fwd * (target_area_frac - frac)))
    status = "arrived" if abs(target_area_frac - frac) < 0.02 and abs(yaw) < 3 else "following"
    return {"vx": round(vx, 1), "yaw_rate_deg": round(yaw, 1), "status": status}
