import math
import os
import sqlite3
from datetime import datetime

import cv2

SEVERITY = {
    "no_helmet": "warning",
    "no_vest": "warning",
    "zone_intrusion": "critical",
}

# The same problem on the SAME worker inside this window is not logged again.
REPEAT_S = {"no_helmet": 30.0, "no_vest": 30.0, "zone_intrusion": 5.0}

# The same problem at the same spot under a NEW worker ID (the tracker lost the
# person) is ignored inside this window, as long as the old ID is gone from the screen.
MERGE_S = {"no_helmet": 60.0, "no_vest": 60.0, "zone_intrusion": 10.0}


class EventManager:
    """
    Turns 'worker X is violating rule Y' (true on every frame) into ONE event
    when the violation starts. Saves a snapshot image and a database row.

    Duplicate protection:
      1) same worker + same problem again inside REPEAT_S  -> ignored (flicker)
      2) same problem at the same spot, and the earlier worker's ID is gone
         from the screen (inside MERGE_S) -> ignored (tracker gave the same
         person a new ID)
    """

    def __init__(self, db_path="data/events.db", snap_dir="data/snapshots",
                 cooldown_s=10.0, merge_distance=150, reset=False):
        os.makedirs(snap_dir, exist_ok=True)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if reset and os.path.exists(db_path):
            os.remove(db_path)

        self.snap_dir = snap_dir
        self.cooldown_s = cooldown_s
        self.merge_distance = merge_distance
        self.active = set()   # (worker_id, type) violations that are on right now
        self.recent = []      # saved events, newest last

        self.db = sqlite3.connect(db_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                video_time REAL,
                type TEXT,
                severity TEXT,
                worker_id INTEGER,
                source TEXT,
                frame_no INTEGER,
                snapshot TEXT
            )""")
        self.db.commit()

    def _is_duplicate(self, worker_id, etype, cx, cy, video_time, visible_ids):
        self.recent = [e for e in self.recent if video_time - e["t"] < 120]
        repeat_s = REPEAT_S.get(etype, 10.0)
        merge_s = MERGE_S.get(etype, self.cooldown_s)
        for e in self.recent:
            if e["type"] != etype:
                continue
            age = video_time - e["t"]
            if e["worker"] == worker_id and age < repeat_s:
                return True
            if (e["worker"] not in visible_ids and age < merge_s
                    and math.hypot(cx - e["cx"], cy - e["cy"]) < self.merge_distance):
                return True
        return False

    def update(self, frame, current, frame_no, video_time, source, visible_ids=()):
        """
        current: {(worker_id, event_type): (x1, y1, x2, y2)} for every
        violation that is true right now. Returns the NEW events saved.
        """
        visible_ids = set(visible_ids)
        new_events = []
        for key, box in current.items():
            if key in self.active:
                continue
            worker_id, etype = key
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if self._is_duplicate(worker_id, etype, cx, cy, video_time, visible_ids):
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.snap_dir, f"{ts}_f{frame_no}_{etype}_w{worker_id}.jpg")
            snap = frame.copy()
            cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(snap, f"{etype.upper()} worker #{worker_id}",
                        (x1, min(y2 + 25, snap.shape[0] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(path, snap)

            severity = SEVERITY.get(etype, "info")
            cur = self.db.execute(
                "INSERT INTO events (ts, video_time, type, severity, worker_id, "
                "source, frame_no, snapshot) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), round(video_time, 2),
                 etype, severity, worker_id, source, frame_no, path))
            self.db.commit()

            self.recent.append({"t": video_time, "type": etype, "worker": worker_id,
                                "cx": cx, "cy": cy})
            new_events.append({"id": cur.lastrowid, "type": etype, "severity": severity,
                               "worker": worker_id, "video_time": video_time})

        self.active = set(current.keys())
        return new_events

    def summary(self):
        return self.db.execute(
            "SELECT type, severity, COUNT(*) FROM events GROUP BY type").fetchall()

    def close(self):
        self.db.close()