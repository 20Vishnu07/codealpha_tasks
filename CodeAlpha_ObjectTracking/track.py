"""Real-time object detection (YOLOv8) + SORT tracking.
Usage:  python track.py                 # webcam
        python track.py --source video.mp4 --save out.mp4"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from sort_tracker import Sort


def color_for(i):
    rng = np.random.default_rng(i * 7919)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam or a video path")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--classes", type=int, nargs="*", help="COCO class ids to keep, e.g. 0 for person")
    ap.add_argument("--save", help="output video path")
    a = ap.parse_args()

    model = YOLO(model=a.model)
    names = model.names
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {a.source}")
    writer = None
    if a.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        size = (int(cap.get(3)), int(cap.get(4)))
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=a.conf, classes=a.classes, verbose=False)[0]
        b = res.boxes
        dets = np.hstack([b.xyxy.cpu().numpy(), b.cls.cpu().numpy()[:, None]]) if len(b) else np.empty((0, 5))
        for tid, box, cls in tracker.update(dets):
            x1, y1, x2, y2 = map(int, box)
            col = color_for(tid)
            label = f"{names[cls]} #{tid}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        if writer:
            writer.write(frame)
        cv2.imshow("Detection + tracking (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
