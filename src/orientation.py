import math


def calculate_orientation(keypoints):

    try:

        nose = keypoints[0]

        left_shoulder = keypoints[5]

        right_shoulder = keypoints[6]

        shoulder_center_x = (
            left_shoulder[0]
            + right_shoulder[0]
        ) / 2

        shoulder_center_y = (
            left_shoulder[1]
            + right_shoulder[1]
        ) / 2

        dx = (
            nose[0]
            - shoulder_center_x
        )

        dy = (
            nose[1]
            - shoulder_center_y
        )

        angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        return angle

    except:

        return None