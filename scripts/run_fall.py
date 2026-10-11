import sys

import cv2
from ultralytics import YOLO

from rules.fall_rules import FallMonitor


def main(source):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter("runs/fall_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    model = YOLO("models/yolo11s-pose.pt")
    monitor = FallMonitor(fps=fps)
    frames = 0
    reported = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        res = model.track(frame, persist=True, conf=0.4,
                          tracker="configs/bytetrack_custom.yaml",
                          device=0, verbose=False)[0]
        annotated = res.plot(boxes=False)   # draws the skeleton

        people = []
        if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
            ids = res.boxes.id.int().cpu().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            kxy = res.keypoints.xy.cpu().numpy()
            kconf = res.keypoints.conf.cpu().numpy()
            for i, tid in enumerate(ids):
                people.append({"id": tid,
                               "box": tuple(int(v) for v in boxes[i]),
                               "kxy": kxy[i], "kconf": kconf[i]})

        report = monitor.update(people)

        banner = None
        for tid, r in report.items():
            x1, y1, x2, y2 = r["box"]
            alert = "FALL" if r["fall"] else ("PERSON DOWN" if r["down"] else None)
            color = (0, 0, 255) if alert else (0, 200, 0)
            angle = "?" if r["angle"] is None else f"{r['angle']:.0f}deg"
            label = f"#{tid} {r['pose']} {angle}" + (f"  {alert}" if alert else "")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (max(x1, 2), max(y1 - 8, 45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if alert:
                banner = alert if banner != "FALL" else banner
                if (tid, alert) not in reported:
                    reported.add((tid, alert))
                    print(f"{alert}: worker #{tid} at frame {frames} ({frames / fps:.1f}s)")

        if banner:
            cv2.rectangle(annotated, (0, 0), (width, 28), (0, 0, 255), -1)
            cv2.putText(annotated, f"{banner} DETECTED", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(annotated)
        cv2.imshow("Fall detection test", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. {frames} frames. Saved to runs\\fall_output.mp4")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0")