"""Build a tiny ONNX model that mimics a YOLO export's input/output contract and returns fixed raw
predictions. The tests use it to exercise the real ONNX Runtime -> postprocess path without
downloading weights. Its output is fake: it is not a detector."""
from __future__ import annotations

import numpy as np


def raw_predictions(num_classes: int = 80, anchors: int = 8400, boxes=None) -> np.ndarray:
    """[1, 4+nc, A] with a few confident boxes (in 640x640 letterboxed pixels) and noise elsewhere."""
    rng = np.random.default_rng(0)
    out = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
    out[0, :4] = rng.uniform(0, 640, (4, anchors))
    out[0, 4:] = rng.uniform(0, 0.05, (num_classes, anchors))  # background
    boxes = boxes or [
        # cx, cy, w, h, class, score
        (320, 320, 100, 200, 0, 0.92),   # person
        (325, 318, 104, 196, 0, 0.80),   # duplicate person -> removed by NMS
        (150, 400, 80, 60, 39, 0.75),    # bottle
        (152, 402, 80, 60, 41, 0.70),    # overlapping cup: different class, kept
        (500, 200, 40, 40, 16, 0.20),    # dog below threshold
    ]
    for i, (cx, cy, w, h, c, s) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


def build(path: str, size: int = 640, num_classes: int = 80, anchors: int = 8400) -> str:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    const = numpy_helper.from_array(raw_predictions(num_classes, anchors), name="raw")
    # output = raw + 0 * mean(images): keeps the graph dependent on the input like a real model
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
        helper.make_node("Mul", ["m", "zero"], ["z"]),
        helper.make_node("Add", ["raw", "z"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + num_classes, anchors])],
        initializer=[const, numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    import sys

    print(build(sys.argv[1] if len(sys.argv) > 1 else "models/fake_yolo.onnx"))
