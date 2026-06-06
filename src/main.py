import cv2

from detector import detect_people
from pose_detector import detect_pose

from interaction import detect_groups
from visualizer import draw

from utils import mask_centroid

from sam2_segmenter import (
    set_frame,
    get_mask
)

from pose_matching import (
    match_pose_to_person
)

from orientation import (
    calculate_orientation
)

from config import *

frame_count = 0

cached_masks = {}
cached_poses = []

cap = cv2.VideoCapture(
    RTSP_URL,
    cv2.CAP_FFMPEG
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not cap.isOpened():

    print("RTSP Connection Failed")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (
            FRAME_WIDTH,
            FRAME_HEIGHT
        )
    )

    frame_count += 1

    # --------------------
    # YOLO Detection
    # --------------------

    detections = detect_people(frame)

    # --------------------
    # YOLO Pose Cache
    # --------------------

    if frame_count % 10 == 0:

        cached_poses = detect_pose(frame)

    poses = cached_poses

    # --------------------
    # SAM2 Update
    # --------------------

    if (
        frame_count % SAM_UPDATE_INTERVAL == 0
        and len(detections) > 0
    ):

        set_frame(
            frame,
            frame_count
        )

    # --------------------
    # Cleanup Old Masks
    # --------------------

    active_ids = {
        det["id"]
        for det in detections
    }

    cached_masks = {
        k: v
        for k, v in cached_masks.items()
        if k in active_ids
    }

    persons = []

    # --------------------
    # Process Persons
    # --------------------

    for det in detections:

        person_id = det["id"]

        x1, y1, x2, y2 = det["box"]

        pose = match_pose_to_person( [x1,y1,x2,y2], poses )
        orientation = None

        if pose is not None:

            orientation = calculate_orientation( pose["keypoints"] )

        width = x2 - x1
        height = y2 - y1

        area = width * height

        mask = None

        try:

            if (
                frame_count % SAM_UPDATE_INTERVAL == 0
                and width > 60
                and height > 60
                and area > 12000
            ):

                mask = get_mask(
                    frame,
                    [x1, y1, x2, y2]
                )

                cached_masks[person_id] = mask

            else:

                mask = cached_masks.get(
                    person_id,
                    None
                )

        except Exception as e:

            print(
                f"SAM2 Error {person_id}: {e}"
            )

        center = None

        if mask is not None:

            center = mask_centroid(mask)

        if center is None:

            center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

        persons.append(
            {
                "id": person_id,
                "box": [x1,y1,x2,y2],
                "center": center,
                "mask": mask,
                "pose": pose,
                "orientation": orientation
            }
        )
    # --------------------
    # Group Detection
    # --------------------

    groups = detect_groups(
        persons
    )

    # --------------------
    # Visualization
    # --------------------

    frame = draw(
        frame,
        persons,
        groups
    )

    cv2.putText(
        frame,
        f"Persons: {len(persons)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Groups: {len(groups)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Group Interaction Tracking",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()