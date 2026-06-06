import math
import time
from interaction_score import ( orientation_score )

from config import (
    GROUP_DISTANCE,
    INTERACTION_TIME,
    MOVEMENT_THRESHOLD,
    ORIENTATION_WEIGHT,
    MOVEMENT_WEIGHT
)

interaction_memory = {}
previous_positions = {}


def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def movement_similarity(v1, v2):

    mag1 = math.sqrt(
        v1[0] ** 2 + v1[1] ** 2
    )

    mag2 = math.sqrt(
        v2[0] ** 2 + v2[1] ** 2
    )

    # both stationary
    if mag1 < 2 and mag2 < 2:
        return 0.3

    if mag1 == 0 or mag2 == 0:
        return 0

    dot = (
        v1[0] * v2[0] +
        v1[1] * v2[1]
    )

    return dot / (mag1 * mag2)


def get_motion(person):

    person_id = person["id"]

    current_center = person["center"]

    if person_id not in previous_positions:

        previous_positions[person_id] = current_center

        return (0, 0)

    prev = previous_positions[person_id]

    vector = (
        current_center[0] - prev[0],
        current_center[1] - prev[1]
    )

    previous_positions[person_id] = current_center

    return vector


def detect_groups(persons):

    groups = []

    current_time = time.time()

    motions = {}

    for person in persons:

        motions[person["id"]] = get_motion(
            person
        )

    for i in range(len(persons)):

        for j in range(i + 1, len(persons)):

            p1 = persons[i]
            p2 = persons[j]

            pair = tuple(
                sorted(
                    [
                        p1["id"],
                        p2["id"]
                    ]
                )
            )

            d = distance(
                p1["center"],
                p2["center"]
            )

            if d > GROUP_DISTANCE:

                if pair in interaction_memory:
                    del interaction_memory[pair]

                continue

            similarity = movement_similarity(
                motions[p1["id"]],
                motions[p2["id"]]
            )

            orientation = orientation_score(
                p1.get("orientation"),
                p2.get("orientation")
            )

            # do NOT reset timer
            combined_score = (
                MOVEMENT_WEIGHT * similarity +
                ORIENTATION_WEIGHT * orientation
            )

            if combined_score < 0.15:
                continue

            if pair not in interaction_memory:

                interaction_memory[pair] = current_time

            duration = (
                current_time -
                interaction_memory[pair]
            )

            if duration >= INTERACTION_TIME:

                groups.append(
                    {
                        "pair": pair,
                        "duration": round(
                            duration,
                            1
                        ),
                        "score": round(
                            combined_score,
                            2
                        )
                    }
                )

    return groups