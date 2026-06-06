def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(
        0,
        xB - xA
    ) * max(
        0,
        yB - yA
    )

    if inter_area == 0:
        return 0

    areaA = (
        boxA[2] - boxA[0]
    ) * (
        boxA[3] - boxA[1]
    )

    areaB = (
        boxB[2] - boxB[0]
    ) * (
        boxB[3] - boxB[1]
    )

    return inter_area / (
        areaA + areaB - inter_area
    )


def match_pose_to_person(
    person_box,
    poses
):

    best_pose = None
    best_iou = 0

    for pose in poses:

        score = iou(
            person_box,
            pose["box"]
        )

        if score > best_iou:

            best_iou = score
            best_pose = pose

    if best_iou < 0.30:
        return None

    return best_pose