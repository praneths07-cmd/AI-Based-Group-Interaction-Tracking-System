# GroupInteractionTracker

GroupInteractionTracker is a Python project for real-time person detection, pose estimation, and group interaction visualization from an RTSP camera feed.

## Features

- Person detection and tracking using YOLO and ByteTrack.
- Human pose estimation and orientation calculation.
- Group interaction detection based on proximity and movement.
- Segmentation masks using SAM2 for more accurate person centroids.
- Overlay visualization with bounding boxes, IDs, orientation, and group lines.

## Requirements

- Python 3.10+ (Python 3.11 recommended)
- GPU recommended for best performance with YOLO and SAM2
- `models/yolov8n.pt` model file present in the repository
- Dependencies installed from `requirements.txt`

## Setup

1. Open a terminal in the repository root:

```powershell
cd D:\texa\GroupInteractionTracker
```

2. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install the required packages:

```powershell
pip install -r requirements.txt
```

## Configuration

Update `src/config.py` with your RTSP camera connection and settings:

```python
RTSP_URL = "rtsp://username:password@camera_ip"
```

You can also adjust:

- `FRAME_WIDTH` and `FRAME_HEIGHT`
- `CONFIDENCE`
- `SAM_UPDATE_INTERVAL`
- `GROUP_DISTANCE`
- `INTERACTION_TIME`

## Run

From the repository root, run:

```powershell
python src/main.py
```

The application reads frames from the RTSP stream, performs detection, pose estimation, mask extraction, and displays results with group interaction overlays.

## Project Structure

- `src/` - Main Python source code
- `models/` - YOLO model files
- `assets/` - Additional resources used by the project
- `logs/` - Output logs
- `requirements.txt` - Python dependencies

## Notes

- The project uses `ultralytics` for YOLO tracking and the SAM2 repository for image segmentation.
- Make sure the model files are available and the RTSP stream is accessible before running.
- For GitHub, commit the README and any project changes together.
