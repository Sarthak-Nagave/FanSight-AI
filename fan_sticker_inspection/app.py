import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO

app = Flask(__name__)

# Configuration
CONFIDENCE_THRESHOLD = 0.70

# Load YOLO Detection Model ONCE at startup
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'fan_sticker_detection', 'weights', 'best.pt'))

if os.path.exists(MODEL_PATH):
    print(f"Loading YOLO Detection model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Detection Model loaded successfully.")
else:
    print(f"ERROR: Model not found at {MODEL_PATH}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        # Read image from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image.'}), 400

        h, w = img.shape[:2]
        source_type = request.form.get('source', 'UNKNOWN')

        # Run YOLO Detection
        results = model(img, verbose=False)
        result = results[0]
        boxes = result.boxes

        valid_detections = []
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # We only care about class 0 (STICKER) and confidence >= threshold
            if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                valid_detections.append({
                    'conf': conf,
                    'x1': float(box.xyxy[0][0]),
                    'y1': float(box.xyxy[0][1]),
                    'x2': float(box.xyxy[0][2]),
                    'y2': float(box.xyxy[0][3])
                })

        # Sort by confidence (highest first)
        valid_detections.sort(key=lambda x: x['conf'], reverse=True)

        print("\n" + "=" * 50)
        print(f"SOURCE: {source_type}")
        print(f"CAPTURE SIZE: {w} x {h}")
        print("MODEL: YOLO11n DETECTION")
        print(f"DETECTIONS: {len(valid_detections)}")

        if len(valid_detections) > 0:
            best_det = valid_detections[0]
            final_result = "PASS"
            
            print("CLASS: STICKER")
            print(f"CONFIDENCE: {best_det['conf'] * 100:.1f}%")
            print(f"BOX: {best_det['x1']:.1f}, {best_det['y1']:.1f}, {best_det['x2']:.1f}, {best_det['y2']:.1f}")
            print(f"RESULT: {final_result}")
            
            response = {
                'result': 'PASS',
                'prediction': 'STICKER DETECTED',
                'confidence': best_det['conf'],
                'box': {
                    'x1': best_det['x1'],
                    'y1': best_det['y1'],
                    'x2': best_det['x2'],
                    'y2': best_det['y2']
                },
                'image_width': w,
                'image_height': h
            }
        else:
            final_result = "FAIL"
            print(f"RESULT: {final_result}")
            
            response = {
                'result': 'FAIL',
                'prediction': 'STICKER NOT DETECTED',
                'confidence': 0.0,
                'box': None,
                'image_width': w,
                'image_height': h
            }
            
        print("=" * 50 + "\n")
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
