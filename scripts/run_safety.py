import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

from detector.ppe_detector import PPEDetector
from events.event_manager import EventManager
from rules.fall_rules import FallMonitor
from rules.ppe_rules import ComplianceMonitor
from rules.zone_rules import ZoneMonitor, load_zones

LABELS = {
    "no_helmet": "NO HELMET",
    "no_vest": "NO VEST",
    "zone_intrusion": "IN RESTRICTED ZONE",
    "fall": "FALL",
    "person_down": "PERSON DOWN",
}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_worker(pose_box, ppe_report, min_iou=0.3):
    """Give a pose-model person the same worker number as the PPE-model person
    that overlaps it most. Returns None if nothing overlaps enough."""
    best_id, best = None, min_iou
    for wid, r in ppe_report.items():
        v = iou(pose_box, r["box"])
        if v > best:
            best_id, best = wid, v
    return best_id


def main(source):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter("runs/safety_output.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (width, height))

    zones = load_zones(frame_size=(width, height))
    zone_polys = [np.array(z["points"], dtype=np.int32) for z in zones]

    detector = PPEDetector()
    pose_model = YOLO("models/yolo11s-pose.pt")
    ppe_monitor = ComplianceMonitor()
    zone_monitor = ZoneMonitor(zones)
    fall_monitor = FallMonitor(fps=src_fps)
    events = EventManager(reset=True)   # reset=True: each test run starts with an empty log

    prev = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        # 1) detect + track (PPE model)
        dets = detector.update(frame)
        persons = [d for d in dets if d.label == "Person" and d.track_id is not None]

        # 2) PPE and zone rules
        ppe_report = ppe_monitor.update(dets)
        zone_status, _ = zone_monitor.update(persons)

        # 3) pose model -> fall / person down
        pose_res = pose_model.track(frame, persist=True, conf=0.4,
                                    tracker="configs/bytetrack_custom.yaml",
                                    device=0, verbose=False)[0]
        people = []
        if (pose_res.boxes is not None and pose_res.boxes.id is not None
                and pose_res.keypoints is not None):
            ids = pose_res.boxes.id.int().cpu().tolist()
            pboxes = pose_res.boxes.xyxy.cpu().numpy()
            kxy = pose_res.keypoints.xy.cpu().numpy()
            kconf = pose_res.keypoints.conf.cpu().numpy()
            for i, pid in enumerate(ids):
                people.append({"id": pid, "box": tuple(int(v) for v in pboxes[i]),
                               "kxy": kxy[i], "kconf": kconf[i]})
        fall_report = fall_monitor.update(people)

        # 4) every violation that is true right now
        current = {}
        for wid, r in ppe_report.items():
            if r["no_helmet"]:
                current[(wid, "no_helmet")] = r["box"]
            if r["no_vest"]:
                current[(wid, "no_vest")] = r["box"]
            if zone_status.get(wid):
                current[(wid, "zone_intrusion")] = r["box"]

        extra = {}   # fall alerts for people the PPE model has no box for
        for pid, r in fall_report.items():
            kind = "fall" if r["fall"] else ("person_down" if r["down"] else None)
            if kind is None:
                continue
            wid = match_worker(r["box"], ppe_report)
            if wid is None:
                wid = 1000 + pid
                extra[wid] = (r["box"], kind)
            current[(wid, kind)] = r["box"]

        # 5) draw the zone
        overlay = frame.copy()
        for poly in zone_polys:
            cv2.fillPoly(overlay, [poly], (0, 0, 255))
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        for poly in zone_polys:
            cv2.polylines(frame, [poly], True, (0, 0, 255), 2)

        # 6) draw workers
        for wid, r in ppe_report.items():
            x1, y1, x2, y2 = r["box"]
            problems = [LABELS[t] for (w, t) in current if w == wid]
            color = (0, 0, 255) if problems else (0, 200, 0)
            label = f"Worker #{wid}: " + (", ".join(problems) if problems else "OK")
            ly = y1 - 8 if y1 > 45 else y1 + 20   # keep the label out of the banner
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (max(x1, 2), ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            fx, fy = int((x1 + x2) / 2), int(y2)
            cv2.circle(frame, (fx, min(fy, height - 3)), 5, color, -1)  # foot point

        for wid, (box, kind) in extra.items():
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Person #{wid}: {LABELS[kind]}", (max(x1, 2), max(y1 + 20, 45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if current:
            cv2.rectangle(frame, (0, 0), (width, 28), (0, 0, 255), -1)
            cv2.putText(frame, "SAFETY ALERT", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 7) save new events (one per violation start) with a snapshot
        visible = set(ppe_report.keys()) | set(extra.keys())
        for e in events.update(frame, current, frames, frames / src_fps, str(source), visible):
            print(f"EVENT #{e['id']} [{e['severity']}] {e['type']} "
                  f"worker #{e['worker']} at {e['video_time']:.1f}s (frame {frames})")

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (width - 110, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        writer.write(frame)
        cv2.imshow("Worker Safety AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {frames} frames. Events saved: {events.summary()}")
    events.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0")