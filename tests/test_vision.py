import io
import math

import numpy as np
import pytest

from vision.processing import Detection, Letterbox, iou, letterbox, nms, postprocess, to_tensor
from vision.tracking import IoUTracker, bearing_deg, distance_m, follow_command


# ---------------------------------------------------------------- pre/post-processing
def test_letterbox_keeps_aspect_and_centres():
    img = np.zeros((720, 1280, 3), np.uint8)
    out, lb = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert lb.scale == 0.5 and lb.pad_x == 0 and lb.pad_y == 140
    assert (out[:140] == 114).all() and (out[140:500] == 0).all() and (out[500:] == 114).all()
    t = to_tensor(out)
    assert t.shape == (1, 3, 640, 640) and t.dtype == np.float32 and t.max() <= 1.0


def test_iou_and_nms():
    a = np.array([0, 0, 10, 10.0])
    bs = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30.0]])
    assert np.allclose(iou(a, bs), [1.0, 25 / 175, 0.0])
    keep = nms(np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60.0]]), np.array([0.6, 0.9, 0.5]), 0.5)
    assert keep == [1, 2]


def test_postprocess_decodes_unletterboxes_and_filters():
    from vision.fake_model import raw_predictions

    lb = Letterbox(scale=0.5, pad_x=0, pad_y=140, orig_w=1280, orig_h=720)
    dets = postprocess(raw_predictions(), lb, labels=[str(i) for i in range(80)], conf=0.25)
    assert [d.class_id for d in dets] == [0, 39, 41]  # duplicate person suppressed, dog below conf
    person = dets[0]
    # letterboxed (270,220)-(370,420) -> original: x/0.5, (y-140)/0.5
    assert (person.x1, person.y1, person.x2, person.y2) == pytest.approx((540, 160, 740, 560))
    assert postprocess(raw_predictions(), lb, conf=0.25, classes={39})[0].class_id == 39
    assert postprocess(raw_predictions(), lb, conf=0.99) == []


def test_postprocess_accepts_transposed_output_and_clips():
    lb = Letterbox(1.0, 0, 0, 100, 100)
    out = np.zeros((1, 6, 20), np.float32)  # 2 classes, 20 anchors (real exports: 84 x 8400)
    out[0, :, 0] = [95, 50, 30, 30, 0.9, 0.0]  # extends past the right edge
    dets_a = postprocess(out, lb)
    dets_b = postprocess(out[0].T[None], lb)  # [1, A, 4+nc] layout
    assert len(dets_a) == len(dets_b) == 1 and dets_a[0].x2 == 100.0


# ---------------------------------------------------------------- real ONNX Runtime path
@pytest.fixture(scope="module")
def fake_model(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from vision.fake_model import build

    return build(str(tmp_path_factory.mktemp("m") / "fake.onnx"))


def test_onnx_detector_end_to_end(fake_model):
    from vision.detector import OnnxDetector

    d = OnnxDetector(fake_model, providers=["CPUExecutionProvider"])
    res = d(np.zeros((720, 1280, 3), np.uint8))
    assert [x.label for x in res.detections] == ["person", "bottle", "cup"]
    assert res.timing.total_ms > 0 and res.image_size == (1280, 720)
    assert res.to_dict()["count"] == 3


def jpeg(w=1280, h=720) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (30, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


def test_api(fake_model):
    from fastapi.testclient import TestClient

    from vision.api import create_app
    from vision.detector import OnnxDetector

    c = TestClient(create_app(OnnxDetector(fake_model, providers=["CPUExecutionProvider"])))
    r = c.post("/detect", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 3
    assert body["detections"][0]["bearing_deg"] == pytest.approx(0.0, abs=0.5)  # person is centred
    r = c.post("/detect/annotated", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    f = c.post("/follow?target=person", files={"file": ("a.jpg", jpeg(), "image/jpeg")}).json()
    assert f["found"] and f["command"]["status"] in {"following", "arrived"}
    assert c.post("/detect", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400
    m = c.get("/metrics").json()
    assert m["requests"] == 3 and m["latency_ms"]["p50"] > 0


def test_bench_runs(fake_model, tmp_path):
    from vision.bench import main

    assert main(["--model", fake_model, "--runs", "5", "--warmup", "1", "--out", str(tmp_path / "b.json")]) == 0
    assert "fps" in (tmp_path / "b.md").read_text()


# ---------------------------------------------------------------- tracking + robot geometry
def det(x, y, w=40, h=80, c=0, s=0.9):
    return Detection(x, y, x + w, y + h, s, c, "person")


def test_tracker_keeps_ids_through_motion_and_a_missed_frame():
    tr = IoUTracker(iou_thres=0.3, max_misses=3, min_hits=1)
    ids = []
    for f in range(12):
        dets = [det(100 + 6 * f, 100), det(400 - 6 * f, 300)]
        if f == 6:
            dets = dets[:1]  # second object missed for one frame
        tr.update(dets)
        ids.append(tuple(d.track_id for d in dets))
    assert ids[0] == (1, 2) and ids[-1] == (1, 2)  # same ids after the gap
    assert len({i for row in ids for i in row}) == 2


def test_tracker_does_not_match_across_classes():
    tr = IoUTracker(min_hits=1)
    a = tr.update([det(100, 100, c=0)])[0].track_id
    b = tr.update([det(100, 100, c=39)])[0].track_id
    assert a != b


def test_bearing_distance_and_follow():
    W, H = 640, 480
    centred = Detection(300, 200, 340, 280, 0.9, 0, "person")
    right = Detection(600, 200, 640, 280, 0.9, 0, "person")
    assert bearing_deg(centred, W) == pytest.approx(0, abs=1e-9)
    assert bearing_deg(right, W) == pytest.approx(math.degrees(math.atan(300 / 320 * math.tan(math.radians(31.1)))), abs=1e-6)
    # a 1.7 m person filling the full frame height is about as far as the camera's vertical view at 1.7 m
    tall = Detection(0, 0, 100, 480, 0.9, 0, "person")
    assert distance_m(tall, H, 1.7) == pytest.approx(1.7 / (2 * math.tan(math.radians(24.4))), rel=1e-6)
    cmd = follow_command(right, W, H)
    assert cmd["yaw_rate_deg"] < 0 and cmd["vx"] > 0  # turn right (clockwise), walk forward
    assert follow_command(None, W, H)["status"] == "searching"
    close = Detection(0, 0, 640, 480, 0.9, 0, "person")
    assert follow_command(close, W, H)["vx"] < 0  # too close: back off
