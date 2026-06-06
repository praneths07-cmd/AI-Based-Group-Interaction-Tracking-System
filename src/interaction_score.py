import math


def angle_difference(a1, a2):

    diff = abs(a1 - a2)

    if diff > 180:
        diff = 360 - diff

    return diff


def orientation_score(angle1, angle2):

    if angle1 is None:
        return 0

    if angle2 is None:
        return 0

    diff = angle_difference(
        angle1,
        angle2
    )

    # Employees facing each other

    if diff >= 120:
        return 1.0

    elif diff >= 90:
        return 0.8

    elif diff >= 60:
        return 0.4

    return 0.0