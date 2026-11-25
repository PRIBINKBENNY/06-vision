"""Explainer figure for the README:  python -m vision.figures  ->  docs/pipeline.png
Shows letterboxing and what class-aware NMS does to raw candidates, using the same synthetic raw
predictions the tests use (not real detections)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .fake_model import raw_predictions
from .processing import Letterbox, letterbox, postprocess


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11})
    W, H = 1280, 720
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (52, 63, 80)
    img[H // 2:] = (74, 88, 104)  # a floor
    boxed, lb = letterbox(img, 640)

    raw = raw_predictions()
    rng = np.random.default_rng(3)
    # add jittered duplicates around each object, like a real detector's neighbouring anchors
    extra = []
    for cx, cy, w, h, c, s in [(320, 320, 100, 200, 0, 0.92), (150, 400, 80, 60, 39, 0.75), (152, 402, 80, 60, 41, 0.70)]:
        for _ in range(7):
            extra.append((cx + rng.normal(0, 6), cy + rng.normal(0, 6), w * rng.uniform(0.9, 1.1),
                          h * rng.uniform(0.9, 1.1), c, s * rng.uniform(0.55, 0.97)))
    raw = raw_predictions(boxes=[(320, 320, 100, 200, 0, 0.92), (325, 318, 104, 196, 0, 0.80),
                                 (150, 400, 80, 60, 39, 0.75), (152, 402, 80, 60, 41, 0.70),
                                 (500, 200, 40, 40, 16, 0.20)] + extra)
    labels = {0: "person", 39: "bottle", 41: "cup", 16: "dog"}
    colors = {0: "#f2a541", 39: "#58c48b", 41: "#5aa0ff", 16: "#ff6b6b"}

    pred = raw[0]
    scores = pred[4:].max(0)
    cls = pred[4:].argmax(0)
    cand = np.nonzero(scores > 0.15)[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.imshow(boxed)
    ax.set_title(f"1 · Letterbox {W}×{H} → 640×640\nscale {lb.scale:g}, grey padding of\n{lb.pad_y:.0f} px top and bottom", loc="left")
    ax.axhline(lb.pad_y, color="white", ls="--", lw=1)
    ax.axhline(640 - lb.pad_y, color="white", ls="--", lw=1)

    ax = axes[1]
    ax.imshow(boxed)
    for i in cand:
        cx, cy, w, h = pred[:4, i]
        ax.add_patch(mp.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, lw=1.2, alpha=0.8,
                                  ec=colors.get(int(cls[i]), "white")))
    ax.set_title(f"2 · Raw candidates above 0.15: {len(cand)}\nneighbouring anchors fire on\nthe same object", loc="left")

    ax = axes[2]
    ax.imshow(img)
    dets = postprocess(raw, Letterbox(lb.scale, lb.pad_x, lb.pad_y, W, H), [labels.get(i, str(i)) for i in range(80)],
                       conf=0.25, iou_thres=0.45)
    placed = []
    for d in dets:
        col = colors[d.class_id]
        ax.add_patch(mp.Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1, fill=False, lw=2.5, ec=col))
        clash = any(abs(d.x1 - x) < 160 and abs(d.y1 - y) < 60 for x, y in placed)
        y = d.y2 + 34 if clash else d.y1 - 10  # put the label under the box if it would hide another
        ax.text(d.x1, y, f"{d.label} {d.score:.2f}", color="black", fontsize=9, bbox=dict(fc=col, ec="none", pad=1.5))
        placed.append((d.x1, d.y1))
    ax.set_title(f"3 · Threshold + class-aware NMS → {len(dets)} detections\nin original {W}×{H} pixels; overlapping\n"
                 f"bottle and cup both survive (different classes)", loc="left")
    for a in axes:
        a.set_xticks([]), a.set_yticks([])
    fig.text(0.01, 0.01, "Synthetic raw predictions (the same ones the tests use), not a real model's output.",
             fontsize=8.5, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/pipeline.png", dpi=100, facecolor="white")
    print("wrote docs/pipeline.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
