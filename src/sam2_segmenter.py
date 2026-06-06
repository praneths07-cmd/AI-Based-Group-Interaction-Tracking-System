import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor

torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision(
    "high"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] SAM2 Device: {DEVICE}")

print("[INFO] Loading SAM2...")

sam_model = build_sam2_hf(
    "facebook/sam2-hiera-base-plus",
    device=DEVICE
)

predictor = SAM2ImagePredictor(
    sam_model
)

print("[INFO] SAM2 Loaded Successfully")

current_frame_id = -1


def set_frame(frame, frame_id):

    global current_frame_id

    if frame_id == current_frame_id:
        return

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    predictor.set_image(
        rgb_frame
    )

    current_frame_id = frame_id

def get_mask(frame, box):

    try:

        input_box = np.array(
            box,
            dtype=np.float32
        )

        masks, scores, logits = predictor.predict(
            box=input_box,
            multimask_output=False
        )

        return masks[0]

    except Exception as e:

        print(
            f"[SAM2 ERROR] {e}"
        )

        return None