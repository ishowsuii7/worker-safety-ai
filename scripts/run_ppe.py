import sys
import time

import cv2

from detector.ppe_detector import PPEDetector
from rules.ppe_rules import ComplianceMonitor


def main(source):
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter("runs/ppe_output.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (width, height))

    detector = PPEDetector()
    monitor = ComplianceMonitor()
    prev = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        report = monitor.update(detector.update(frame))

        for wid, r in report.items():
            x1, y1, x2, y2 = r["box"]
            problems = []
            if r["no_helmet"]:
                problems.append("NO HELMET")
            if r["no_vest"]:
                problems.append("NO VEST")
            color = (0, 0, 255) if problems else (0, 200, 0)
            label = f"Worker #{wid}: " + (", ".join(problems) if problems else "OK")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        writer.write(frame)
        cv2.imshow("Worker Safety AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frames} frames. Saved to runs\\ppe_output.mp4")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0")