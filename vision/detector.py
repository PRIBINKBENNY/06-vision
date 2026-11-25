"""ONNX Runtime YOLO detector with a per-stage timing breakdown."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .processing import Detection, letterbox, postprocess, to_tensor

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote "
        "keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
        "hair_drier toothbrush").split()


@dataclass
class Timing:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass
class Result:
    detections: list[Detection]
    timing: Timing
    image_size: tuple[int, int]
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "count": len(self.detections),
            "image": {"width": self.image_size[0], "height": self.image_size[1]},
            "timing_ms": {"preprocess": round(self.timing.preprocess_ms, 2),
                          "inference": round(self.timing.inference_ms, 2),
                          "postprocess": round(self.timing.postprocess_ms, 2),
                          "total": round(self.timing.total_ms, 2)},
            **self.extra,
        }


class OnnxDetector:
    def __init__(self, model_path: str, labels: list[str] | None = None, providers: list[str] | None = None,
                 conf: float = 0.25, iou: float = 0.45, threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        available = ort.get_available_providers()
        providers = providers or [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                                              "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else 640
        self.labels = list(labels) if labels else list(COCO)
        self.conf, self.iou = conf, iou
        self.provider = self.session.get_providers()[0]

    def __call__(self, image: np.ndarray, conf: float | None = None, classes: set[int] | None = None) -> Result:
        t0 = time.perf_counter()
        boxed, lb = letterbox(image, self.size)
        x = to_tensor(boxed)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0]
        t2 = time.perf_counter()
        dets = postprocess(out, lb, self.labels, conf if conf is not None else self.conf, self.iou, classes=classes)
        t3 = time.perf_counter()
        return Result(dets, Timing((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3), (lb.orig_w, lb.orig_h))


def load_image(data: bytes) -> np.ndarray:
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.asarray(img)


def draw(image: np.ndarray, dets: list[Detection]) -> bytes:
    """Annotated JPEG bytes."""
    import io

    from PIL import Image, ImageDraw

    im = Image.fromarray(image)
    d = ImageDraw.Draw(im)
    palette = [(242, 165, 65), (88, 196, 139), (90, 160, 255), (255, 107, 107), (200, 140, 255)]
    for det in dets:
        col = palette[det.class_id % len(palette)]
        d.rectangle([det.x1, det.y1, det.x2, det.y2], outline=col, width=3)
        tag = f"{det.label} {det.score:.2f}" + (f" #{det.track_id}" if det.track_id is not None else "")
        tw = d.textlength(tag)
        d.rectangle([det.x1, max(0, det.y1 - 16), det.x1 + tw + 8, max(16, det.y1)], fill=col)
        d.text((det.x1 + 4, max(0, det.y1 - 15)), tag, fill=(0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()
