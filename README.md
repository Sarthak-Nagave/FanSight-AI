# FanSight-AI

AI-powered industrial vision inspection system for detecting the presence of a required sticker on a fan using YOLO object detection.

## Overview

FanSight-AI is a computer vision-based inspection system developed for industrial fan sticker inspection.

The system uses a YOLO11n object detection model to identify the sticker on the fan and provides a simple inspection decision:

- **PASS** — Sticker detected
- **FAIL** — Sticker not detected

When the sticker is detected, the application displays the actual detected sticker location using a bounding box.

The system supports both image upload and camera-based inspection.

## Key Features

- Fan sticker presence detection
- YOLO11n object detection
- Image upload inspection
- Camera capture inspection
- Automatic PASS/FAIL decision
- Actual sticker bounding-box visualization
- Detection confidence display
- Flask backend
- Web-based industrial inspection interface
- Fast local inference

## Inspection Logic

The inspection logic is intentionally simple.

```text
Fan Image
    |
    v
YOLO11n Detection
    |
    v
Sticker Detected?
   /        \
 Yes        No
  |          |
 PASS       FAIL
  |
Actual Sticker
Bounding Box
```

### PASS Condition

If the YOLO model detects the `STICKER` class with a confidence score greater than or equal to the configured threshold:

```text
Result: PASS
Prediction: STICKER DETECTED
```

The actual YOLO bounding box is displayed around the detected sticker.

### FAIL Condition

If no valid sticker detection is found:

```text
Result: FAIL
Prediction: STICKER NOT DETECTED
```

No artificial bounding box is displayed.

## Technology Stack

- Python
- Flask
- YOLO11n
- Ultralytics
- OpenCV
- HTML
- CSS
- JavaScript

## Model

The project uses a YOLO11n object detection model trained for a single class:

```text
Class 0: STICKER
```

The model performs object detection rather than whole-image classification. This allows the system to both determine whether the sticker is present and return its actual location in the image.

## Application Workflow

### Image Upload

```text
Select Image
     |
     v
Preview Image
     |
     v
Run Inspection
     |
     v
YOLO11n Detection
     |
     v
PASS / FAIL
```

### Camera Inspection

```text
Open Camera
     |
     v
Capture Frame
     |
     v
Run Inspection
     |
     v
YOLO11n Detection
     |
     v
PASS / FAIL
```

## Detection Threshold

The current production confidence threshold is:

```text
0.70
```

Only sticker detections meeting or exceeding this threshold are considered valid for the inspection decision.

## Project Structure

```text
FanSight-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── test_detection.py
│   └── ...
│
├── templates/
│   └── ...
│
└── static/
    ├── css/
    └── js/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Sarthak-Nagave/FanSight-AI.git
```

Navigate to the project directory:

```bash
cd FanSight-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Flask application:

```bash
python app.py
```

Open the local application URL displayed by Flask in your browser.

## Testing

The system has been tested with:

- Sticker-present images
- Sticker-missing images
- Image upload inspection
- Camera capture inspection
- YOLO sticker detection
- Actual sticker bounding-box visualization
- PASS/FAIL result generation

The inspection pipeline has been verified for both sticker-present and sticker-missing conditions.

## Expected Results

### Sticker Present

```text
Result: PASS
Prediction: STICKER DETECTED
Confidence: Model confidence score
Bounding Box: Actual sticker location
```

### Sticker Missing

```text
Result: FAIL
Prediction: STICKER NOT DETECTED
Confidence: 0
Bounding Box: None
```

## Model and Dataset Files

Large model files, datasets, training outputs, captured images, and temporary files are excluded from the Git repository through `.gitignore`.

The trained model weights should be placed in the expected model directory before running the application.

## Production Design

The production inference pipeline uses the YOLO object detection model directly.

The following are not used as fallback detection methods:

- Image classification model
- Fixed sticker ROI
- Artificial bounding boxes
- OpenCV-based fake sticker detection

The bounding box shown in the application is generated directly from the YOLO detection result.

## Project Status

**Status: Successfully implemented and tested**

The FanSight-AI POC provides a functional fan sticker presence inspection workflow with image upload, camera capture, YOLO-based detection, PASS/FAIL decision-making, and actual sticker localization.

## Future Improvements

Potential future improvements include:

- Additional real-world camera samples
- Improved robustness across different lighting conditions
- Additional fan models and variations
- Industrial camera integration
- Production hardware integration
- Inspection result logging
- PLC integration
- Deployment optimization

## Author
Sarthak Nagave

## License

This project is developed as an industrial computer vision proof of concept.
```