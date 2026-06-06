import os

import cv2
from ultralytics import YOLO

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "yolov8n-pose.pt"
)

pose_model = YOLO(MODEL_PATH)


def detect_pose(frame):

    small_frame = cv2.resize(
        frame,
        (640, 360)
    )

    results = pose_model(
        small_frame,
        imgsz=640,
        conf=0.5,
        verbose=False
    )

    poses = []

    if len(results) == 0:
        return poses

    result = results[0]

    if result.keypoints is None:
        return poses

    boxes = result.boxes.xyxy.cpu().numpy()

    keypoints = result.keypoints.xy.cpu().numpy()

    for box, points in zip(
        boxes,
        keypoints
    ):

        poses.append(
            {
                "box": box,
                "keypoints": points
            }
        )

    return poses