import os
from ultralytics import YOLO

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "yolov8n.pt"
)

model = YOLO(MODEL_PATH)


def detect_people(frame):

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.4,
        iou=0.5,
        imgsz=640,
        verbose=False
    )

    detections = []

    if len(results) == 0:
        return detections

    result = results[0]

    if result.boxes is None:
        return detections

    if result.boxes.id is None:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.cpu().numpy()

    for box, track_id in zip(boxes, ids):

        x1, y1, x2, y2 = map(int, box)

        detections.append(
            {
                "id": int(track_id),
                "box": [x1, y1, x2, y2]
            }
        )

    return detections