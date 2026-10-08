import sys

import cv2
import yaml

points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def main(source, frame_no=0):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read a frame from:", source)
        return

    h, w = frame.shape[:2]
    win = "Draw restricted zone"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Left-click = add corner | u = undo | Enter = save | q = quit")

    while True:
        img = frame.copy()
        for p in points:
            cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
        if len(points) >= 2:
            for a, b in zip(points, points[1:]):
                cv2.line(img, tuple(a), tuple(b), (0, 0, 255), 2)
        if len(points) >= 3:
            overlay = img.copy()
            cv2.fillPoly(overlay, [cv2.UMat(0) and None or __import__("numpy").array(points)], (0, 0, 255))
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        cv2.imshow(win, img)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("q"):
            print("Cancelled, nothing saved.")
            break
        elif key in (13, 10):
            if len(points) < 3:
                print("Need at least 3 points.")
                continue
            cfg = {
                "frame_size": [w, h],
                "zones": [{"name": "restricted_zone_1", "points": points}],
            }
            with open("configs/zones.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print("Saved configs\\zones.yaml with", len(points), "points")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1]
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(src, frame)