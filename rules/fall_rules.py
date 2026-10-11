import math
from collections import defaultdict

# COCO keypoint indexes used by YOLO pose models
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
MIN_KP_CONF = 0.4


def _mid(kxy, kconf, a, b):
    pts = [kxy[i] for i in (a, b) if kconf[i] >= MIN_KP_CONF]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def torso_angle(kxy, kconf):
    """Angle of the shoulders->hips line from vertical, in degrees.
    0 = standing straight, 90 = lying flat. None if the points are not visible."""
    shoulders = _mid(kxy, kconf, L_SHOULDER, R_SHOULDER)
    hips = _mid(kxy, kconf, L_HIP, R_HIP)
    if shoulders is None or hips is None:
        return None
    dx = hips[0] - shoulders[0]
    dy = hips[1] - shoulders[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def posture(box, kxy, kconf):
    """Returns (state, angle). state is 'upright', 'lying' or 'unknown'.
    If the shoulders and hips are not both visible we do NOT guess."""
    angle = torso_angle(kxy, kconf)
    if angle is None:
        return "unknown", None

    x1, y1, x2, y2 = box
    aspect = (x2 - x1) / max(y2 - y1, 1)   # width / height of the person box
    if angle > 55 and aspect > 1.0:
        return "lying", angle
    if angle < 35:
        return "upright", angle
    return "unknown", angle                # bending, sitting, leaning


class FallMonitor:
    """
    Two different alerts:

    fall : the person was 'upright' and then stayed 'lying' for `lying_seconds`
           (the transition upright -> lying happened no more than
           `transition_seconds` before the lying started).
    down : the person has been 'lying' for `down_seconds`, even if we never
           saw them fall (they may have fallen before the camera saw them).
    """

    def __init__(self, fps=25, lying_seconds=2.0, transition_seconds=3.0, down_seconds=5.0):
        self.lying_needed = max(1, int(lying_seconds * fps))
        self.down_needed = max(1, int(down_seconds * fps))
        self.transition_max = int(transition_seconds * fps)
        self.state = defaultdict(lambda: {"lying": 0, "since_upright": 10 ** 9})

    def update(self, people):
        """people: list of {'id', 'box', 'kxy', 'kconf'}. Returns {id: info}."""
        report = {}
        for p in people:
            st = self.state[p["id"]]
            pose, angle = posture(p["box"], p["kxy"], p["kconf"])

            if pose == "upright":
                st["since_upright"] = 0
                st["lying"] = 0
            else:
                st["since_upright"] += 1
                if pose == "lying":
                    st["lying"] += 1

            gap = st["since_upright"] - st["lying"]   # frames between upright and lying
            fall = st["lying"] >= self.lying_needed and gap <= self.transition_max
            down = st["lying"] >= self.down_needed
            report[p["id"]] = {"box": p["box"], "pose": pose, "angle": angle,
                               "fall": fall, "down": down}
        return report