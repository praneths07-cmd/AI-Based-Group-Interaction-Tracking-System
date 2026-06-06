import cv2
import numpy as np

def mask_centroid(mask):

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    M = cv2.moments(largest)

    if M["m00"] == 0:
        return None

    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])

    return (cx,cy)

import math


def body_orientation(keypoints):

    try:

        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        cx = (
            left_shoulder[0] +
            right_shoulder[0]
        ) / 2

        cy = (
            left_shoulder[1] +
            right_shoulder[1]
        ) / 2

        nose = keypoints[0]

        dx = nose[0] - cx
        dy = nose[1] - cy

        angle = math.atan2(
            dy,
            dx
        )

        return angle

    except:
        return None