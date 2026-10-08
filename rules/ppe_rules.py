from collections import defaultdict, deque


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def inside(point, region):
    x, y = point
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def overlap_ratio(box, region):
    """Fraction of `box` that lies inside `region`."""
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    return inter / area if area > 0 else 0.0


def head_region(person_box):
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return (x1, y1 - 0.05 * h, x2, y1 + 0.30 * h)


def torso_region(person_box):
    # Taller than before: people close to the camera are cut off by the frame,
    # so their torso reaches further down the box.
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return (x1, y1 + 0.10 * h, x2, y1 + 0.85 * h)


def assign(detections, persons, region_fn, min_overlap=0.5):
    """Give each detection to the ONE person it overlaps most.
    Returns {track_id: [detections]}."""
    owned = defaultdict(list)
    for d in detections:
        best_id, best = None, min_overlap
        for p in persons:
            r = overlap_ratio(d.box, region_fn(p.box))
            if r >= best:
                best_id, best = p.track_id, r
        if best_id is not None:
            owned[best_id].append(d)
    return owned


class WorkerState:
    """What we remember about one tracked worker."""

    def __init__(self, helmet_window):
        self.frames_since_vest = 0
        self.helmet_hist = deque(maxlen=helmet_window)


class ComplianceMonitor:
    """
    Turns per-frame detections into stable violations.

    NO VEST   : no vest seen on this worker for `vest_grace` frames in a row.
                (72 frames = 3 seconds at 24 FPS). Any sighting resets the clock.
    NO HELMET : a 'Non-Helmet' seen on the head in at least `helmet_ratio`
                of the last `helmet_window` frames.
    """

    def __init__(self, vest_grace=72, helmet_window=30, helmet_ratio=0.8):
        self.vest_grace = vest_grace
        self.helmet_window = helmet_window
        self.helmet_ratio = helmet_ratio
        self.state = defaultdict(lambda: WorkerState(helmet_window))

    def update(self, detections):
        persons = [d for d in detections if d.label == "Person" and d.track_id is not None]
        helmets = [d for d in detections if d.label == "Helmet"]
        bare_heads = [d for d in detections if d.label == "Non-Helmet"]
        vests = [d for d in detections if d.label == "Vest"]

        own_helmet = assign(helmets, persons, head_region)
        own_bare = assign(bare_heads, persons, head_region)
        own_vest = assign(vests, persons, torso_region)

        report = {}
        for p in persons:
            helmet = bool(own_helmet.get(p.track_id))
            non_helmet = bool(own_bare.get(p.track_id))
            vest = bool(own_vest.get(p.track_id))
            s = self.state[p.track_id]

            if vest:
                s.frames_since_vest = 0
            else:
                s.frames_since_vest += 1

            if helmet or non_helmet:   # only judge when the head is clearly seen
                s.helmet_hist.append(non_helmet and not helmet)

            report[p.track_id] = {
                "box": p.box,
                "no_helmet": self._helmet_violation(s),
                "no_vest": s.frames_since_vest >= self.vest_grace,
            }
        return report

    def _helmet_violation(self, s):
        hist = s.helmet_hist
        return len(hist) == self.helmet_window and sum(hist) / self.helmet_window >= self.helmet_ratio