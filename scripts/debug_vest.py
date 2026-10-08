import sys

import cv2
from ultralytics import YOLO

from rules.ppe_rules import torso_region, center, inside


def main(source, target=200):
    model = YOLO("models/ppe_best.pt")
    cap = cv2.VideoCapture(source)
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        # very low confidence so we can see everything the model considers
        res = model.track(frame, persist=True, conf=0.05,
                          tracker="bytetrack.yaml", device=0, verbose=False)[0]

        if n % 20 != 0 and n != target:
            continue

        persons, vests = [], []
        for i, b in enumerate(res.boxes):
            label = model.names[int(b.cls)]
            conf = float(b.conf)
            box = tuple(int(v) for v in b.xyxy[0].tolist())
            tid = int(b.id) if b.id is not None else None
            if label == "Person" and conf >= 0.4:
                persons.append((tid, box, conf))
            elif label == "Vest":
                vests.append((box, conf))

        print(f"\n--- frame {n} ---")
        print(f"vests seen (any confidence): {len(vests)}")
        for box, conf in vests:
            print(f"   vest conf={conf:.2f} center={tuple(int(v) for v in center(box))}")
        for tid, box, conf in persons:
            torso = torso_region(box)
            matched = [round(c, 2) for vb, c in vests if inside(center(vb), torso)]
            print(f"   person id={tid} box={box} torso_y=({int(torso[1])},{int(torso[3])}) "
                  f"vests matched in torso: {matched}")

        if n == target:
            img = frame.copy()
            for tid, box, conf in persons:
                cv2.rectangle(img, box[:2], box[2:], (0, 255, 0), 2)
                t = torso_region(box)
                cv2.rectangle(img, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (255, 0, 0), 2)
                cv2.putText(img, f"id {tid}", (box[0], box[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for box, conf in vests:
                cv2.rectangle(img, box[:2], box[2:], (0, 255, 255), 2)
                cv2.putText(img, f"vest {conf:.2f}", (box[0], box[3] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(f"runs/debug_frame{target}.png", img)
            print(f"\nSaved runs\\debug_frame{target}.png")
            break

    cap.release()


if __name__ == "__main__":
    main(sys.argv[1])