import cv2
import numpy as np


def draw(frame, persons, groups):

    overlay = frame.copy()

    for person in persons:

        x1, y1, x2, y2 = person["box"]

        mask = person.get("mask")

        if mask is not None:

            mask = mask.astype(np.uint8)

            if mask.shape[:2] != frame.shape[:2]:

                mask = cv2.resize(
                    mask,
                    (
                        frame.shape[1],
                        frame.shape[0]
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

            overlay[mask > 0] = (
                255,
                0,
                255
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            person["center"],
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"ID {person['id']}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        if person.get("orientation") is not None:

            cv2.putText(
                frame,
                f"A:{int(person['orientation'])}",
                (
                    x1,
                    y2 + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

    frame = cv2.addWeighted(
        frame,
        0.7,
        overlay,
        0.3,
        0
    )

    person_map = {
        p["id"]: p
        for p in persons
    }

    for group in groups:

        id1, id2 = group["pair"]

        if id1 not in person_map:
            continue

        if id2 not in person_map:
            continue

        p1 = person_map[id1]
        p2 = person_map[id2]

        cv2.line(
            frame,
            p1["center"],
            p2["center"],
            (255, 0, 0),
            3
        )

        mx = (
            p1["center"][0] +
            p2["center"][0]
        ) // 2

        my = (
            p1["center"][1] +
            p2["center"][1]
        ) // 2

        cv2.putText(
            frame,
            f"GROUP {group['duration']}s",
            (mx, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return frame
