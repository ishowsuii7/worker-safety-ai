import time

import yaml
from shapely.geometry import Point, Polygon


def load_zones(path="configs/zones.yaml", frame_size=None):
    """Loads zones and scales them if the video size differs from the drawing size."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    sx = sy = 1.0
    if frame_size is not None:
        sx = frame_size[0] / cfg["frame_size"][0]
        sy = frame_size[1] / cfg["frame_size"][1]

    zones = []
    for z in cfg["zones"]:
        pts = [(x * sx, y * sy) for x, y in z["points"]]
        zones.append({"name": z["name"], "points": pts, "polygon": Polygon(pts).buffer(6)})
    return zones


def foot_point(box):
    """Bottom-center of the person box = where the worker stands."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


class ZoneMonitor:
    """
    A worker is 'in the zone' after being inside for `enter_frames` frames in a row,
    and leaves it after being outside for `exit_frames` frames in a row.
    One event is created per entry (not one per frame).
    """

    def __init__(self, zones, enter_frames=12, exit_frames=12):
        self.zones = zones
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.state = {}  # (track_id, zone_name) -> counters

    def update(self, persons):
        """persons: Detection objects with label 'Person' and a track_id."""
        status = {}   # track_id -> list of zone names the worker is currently in
        events = []   # new entries this frame

        for p in persons:
            pt = Point(foot_point(p.box))
            active_zones = []
            for z in self.zones:
                key = (p.track_id, z["name"])
                st = self.state.setdefault(key, {"inside": 0, "outside": 0, "active": False})

                if z["polygon"].contains(pt):
                    st["inside"] += 1
                    st["outside"] = 0
                else:
                    st["outside"] += 1
                    st["inside"] = 0

                if not st["active"] and st["inside"] >= self.enter_frames:
                    st["active"] = True
                    events.append({
                        "time": time.strftime("%H:%M:%S"),
                        "type": "zone_intrusion",
                        "worker": p.track_id,
                        "zone": z["name"],
                    })
                elif st["active"] and st["outside"] >= self.exit_frames:
                    st["active"] = False

                if st["active"]:
                    active_zones.append(z["name"])
            status[p.track_id] = active_zones

        return status, events