# Fan Sticker Presence Inspection

## 1. Project Purpose
This project is a computer-vision Proof of Concept (POC) for detecting whether an electric fan has its center sticker (label) present or missing. It uses YOLO11n Classification (YOLO11n-cls) to classify images into `STICKER_PRESENT` or `STICKER_MISSING`.

The model is rotation-invariant, meaning it learns to classify the presence or absence of the sticker regardless of how the fan is rotated, provided there is variation in the dataset.

## 2. Dataset Structure
The dataset is structured as an image classification directory format:
```
dataset/
├── train/
│   ├── STICKER_PRESENT/
│   └── STICKER_MISSING/
└── val/
    ├── STICKER_PRESENT/
    └── STICKER_MISSING/
```

During initialization, we observed the original dataset (`Fan_Project_Dataset`) contained:
- `With_Sticker_Imgs`: 48 images
- `No_Sticker_Imgs`: 18 images
All images were `.jpg` files. No corrupted files were detected.

## 3. Installation
Requirements: Python 3.8+ (Windows supported).

```cmd
cd fan_sticker_inspection
pip install -r requirements.txt
```

## 4. Dataset Preparation
The dataset has already been prepared via the `prepare_dataset.py` script. To redo the preparation:
```cmd
python prepare_dataset.py
```

## 5. Training
To train the YOLO11n classification model on the dataset:
```cmd
cd fan_sticker_inspection
python src/train.py
```
This will automatically download the pre-trained `yolo11n-cls.pt` and fine-tune it. The best model will be saved at `runs/train/fan_sticker_cls/weights/best.pt`.

## 6. Validation
To evaluate the model and see top-1 accuracy:
```cmd
python src/validate.py
```
Check `runs/val/fan_sticker_val` for confusion matrix and other metrics to ensure classes are not being confused.

## 7. Single-image Prediction
You can test a single image from the command line:
```cmd
python src/predict.py path\to\image.jpg
```
Output will specify Prediction (Class), Confidence (%), and Result (PASS/FAIL/UNCERTAIN).

## 8. Webcam Inspection
For real-time testing using a connected webcam:
```cmd
python src/webcam.py
```
Press `q` to quit the webcam feed.

## 9. Streamlit Application
A simple testing UI for image uploads and live webcam (experimental):
```cmd
streamlit run app.py
```

## 10. How to change confidence threshold
Open `config.yaml` and change `confidence_threshold` under `inference`. Alternatively, you can change this dynamically in the Streamlit UI. If the model's confidence is below the threshold, the output is `UNCERTAIN`.

## 11. Troubleshooting
- **Model not found:** Ensure you run `python src/train.py` before validation, prediction, or running the UI.
- **Webcam not opening:** Check if `camera_id=0` is correct in `src/webcam.py` and ensure the camera isn't used by another application.
- **CUDA errors:** Ensure you have the appropriate NVIDIA drivers and PyTorch version installed for GPU support. If not, training will automatically fall back to CPU.
