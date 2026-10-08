from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class Track:
    track_id: int
    box: tuple  # (x1, y1, x2, y2) in pixels
    conf: float


class PersonTracker:
    """Detects people and gives each one a persistent ID (YOLO + ByteTrack)."""

    def __init__(self, model_path="models/yolo11n.pt", conf=0.4, device=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def update(self, frame):
        """Process one video frame and return a list of Track objects."""
        results = self.model.track(
            frame,
            persist=True,           # keep IDs between frames
            classes=[0],            # class 0 = person
            conf=self.conf,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        tracks = []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
            tracks.append(Track(tid, (int(x1), int(y1), int(x2), int(y2)), float(c)))
        return tracks