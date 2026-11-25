"""Latency benchmark: preprocess / inference / postprocess, p50/p95, FPS, across input sizes and
execution providers. Writes a markdown table you can paste into the README.

    python -m vision.bench --model models/yolo11n.onnx --runs 100
    python -m vision.bench --model models/yolo11n_320.onnx --model models/yolo11n.onnx --label "Raspberry Pi 5"
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import numpy as np

from .detector import OnnxDetector


def pct(v: list[float], p: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def bench(model: str, runs: int, warmup: int, image: np.ndarray, providers=None, threads=None) -> dict:
    d = OnnxDetector(model, providers=providers, threads=threads)
    for _ in range(warmup):
        d(image)
    pre, inf, post, tot = [], [], [], []
    for _ in range(runs):
        r = d(image)
        pre.append(r.timing.preprocess_ms)
        inf.append(r.timing.inference_ms)
        post.append(r.timing.postprocess_ms)
        tot.append(r.timing.total_ms)
    return {"model": Path(model).name, "input": d.size, "provider": d.provider.replace("ExecutionProvider", ""),
            "pre_ms": round(statistics.median(pre), 2), "inf_ms": round(statistics.median(inf), 2),
            "post_ms": round(statistics.median(post), 2), "p50_ms": round(pct(tot, 50), 2),
            "p95_ms": round(pct(tot, 95), 2), "fps": round(1000 / statistics.median(tot), 1)}


def table(rows: list[dict], label: str) -> str:
    head = f"**{label}** ({platform.processor() or platform.machine()}, {platform.system()})\n\n"
    cols = ["model", "input", "provider", "pre_ms", "inf_ms", "post_ms", "p50_ms", "p95_ms", "fps"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * 3 + "---:|" * (len(cols) - 3)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return head + "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--image", help="test image (default: random noise at 1280x720)")
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--provider", action="append", help="e.g. CPUExecutionProvider")
    p.add_argument("--threads", type=int)
    p.add_argument("--label", default="this machine")
    p.add_argument("--out", default="results/bench.json")
    a = p.parse_args(argv)
    if a.image:
        from .detector import load_image

        img = load_image(Path(a.image).read_bytes())
    else:
        img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    rows = [bench(m, a.runs, a.warmup, img, a.provider, a.threads) for m in a.model]
    md = table(rows, a.label)
    print(md)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"label": a.label, "rows": rows}, indent=2))
    Path(a.out).with_suffix(".md").write_text(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
