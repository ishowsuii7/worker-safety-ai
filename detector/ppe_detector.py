from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    box: tuple  # (x1, y1, x2, y2)
    conf: float
    track_id: int | None = None


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _inter(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _iou(a, b):
    inter = _inter(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def clean_persons(persons):
    """
    Removes two kinds of bad Person boxes:
      1) GROUP boxes: one big box that contains two or more smaller person boxes.
      2) DUPLICATE boxes: two boxes on the same person (the older track ID wins).
    """
    # 1) group boxes
    keep = []
    for a in persons:
        inside = 0
        for b in persons:
            if b is a:
                continue
            smaller = _area(b.box) < 0.8 * _area(a.box)
            mostly_inside = _inter(a.box, b.box) / max(_area(b.box), 1) > 0.7
            if smaller and mostly_inside:
                inside += 1
        if inside < 2:
            keep.append(a)

    # 2) duplicates: lower (older) track ID is kept
    keep.sort(key=lambda d: (d.track_id is None, d.track_id or 0))
    final = []
    for d in keep:
        if all(_iou(d.box, k.box) < 0.6 for k in final):
            final.append(d)
    return final


class PPEDetector:
    """Runs the trained PPE model with tracking and returns labeled detections."""

    def __init__(self, model_path="models/ppe_best.pt", device=0):
        self.model = YOLO(model_path)
        self.device = device
        # Low base threshold, then each class is filtered by its own threshold.
        # Vest is lower because the model often misses hi-vis jackets.
        self.conf = 0.25
        self.class_conf = {
            "Person": 0.4,
            "Helmet": 0.4,
            "Non-Helmet": 0.4,
            "Vest": 0.25,
        }

    def update(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            tracker="configs/bytetrack_custom.yaml",
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
        else:
            ids = [None] * len(classes)

        out = []
        for (x1, y1, x2, y2), c, cf, tid in zip(xyxy, classes, confs, ids):
            label = self.model.names[c]
            if cf < self.class_conf.get(label, 0.4):
                continue
            out.append(Detection(label,
                                 (int(x1), int(y1), int(x2), int(y2)),
                                 float(cf), tid))

        persons = clean_persons([d for d in out if d.label == "Person"])
        others = [d for d in out if d.label != "Person"]
        return persons + others