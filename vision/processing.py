"""YOLO pre/post-processing in plain NumPy: letterbox, decode, NMS, un-letterbox.

Works with YOLOv8 / YOLO11 ONNX exports (`yolo export model=yolo11n.pt format=onnx`), whose output is
[1, 4 + num_classes, num_anchors]: cx, cy, w, h (input pixels) followed by per-class scores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str = ""
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("x1", "y1", "x2", "y2"):
            d[k] = round(d[k], 1)
        d["score"] = round(d["score"], 3)
        return d


@dataclass
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int


def letterbox(img: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio, pad to size x size. img: HxWx3 uint8 RGB."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        from PIL import Image

        img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), color, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py : py + nh, px : px + nw] = img
    return out, Letterbox(r, px, py, w, h)


def to_tensor(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> 1x3xHxW float32 in [0, 1]."""
    return np.ascontiguousarray(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou(boxes[i], boxes[rest]) <= iou_thres]
    return keep


def postprocess(output: np.ndarray, lb: Letterbox, labels: list[str] | None = None, conf: float = 0.25,
                iou_thres: float = 0.45, max_det: int = 100, classes: set[int] | None = None) -> list[Detection]:
    pred = output[0] if output.ndim == 3 else output  # [4+nc, A]
    if pred.shape[0] > pred.shape[1]:                  # some exports are already [A, 4+nc]
        pred = pred.T
    boxes_cxcywh, scores_all = pred[:4].T, pred[4:].T  # [A,4], [A,nc]
    class_ids = scores_all.argmax(1)
    scores = scores_all[np.arange(len(class_ids)), class_ids]
    mask = scores >= conf
    if classes is not None:
        mask &= np.isin(class_ids, list(classes))
    if not mask.any():
        return []
    b, s, c = boxes_cxcywh[mask], scores[mask], class_ids[mask]
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    # class-aware NMS in one pass: shift each class's boxes far apart so they never overlap
    offset = c[:, None].astype(np.float32) * 10_000
    keep = nms(xyxy + offset, s, iou_thres)[:max_det]

    dets = []
    for i in keep:
        x1, y1, x2, y2 = xyxy[i]
        # undo the letterbox: remove padding, divide by scale, clip to the original image
        x1 = float(np.clip((x1 - lb.pad_x) / lb.scale, 0, lb.orig_w))
        x2 = float(np.clip((x2 - lb.pad_x) / lb.scale, 0, lb.orig_w))
        y1 = float(np.clip((y1 - lb.pad_y) / lb.scale, 0, lb.orig_h))
        y2 = float(np.clip((y2 - lb.pad_y) / lb.scale, 0, lb.orig_h))
        cid = int(c[i])
        dets.append(Detection(x1, y1, x2, y2, float(s[i]), cid, labels[cid] if labels and cid < len(labels) else str(cid)))
    return dets
