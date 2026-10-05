import os
import sys
import argparse
from ultralytics import YOLO

def test_image(model_path, image_path, conf_threshold=0.70):
    print(f"\nImage:\n{os.path.basename(image_path)}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, verbose=False)
    result = results[0]
    
    boxes = result.boxes
    
    # Filter detections by confidence threshold
    valid_detections = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            valid_detections.append({
                'conf': conf,
                'x1': float(box.xyxy[0][0]),
                'y1': float(box.xyxy[0][1]),
                'x2': float(box.xyxy[0][2]),
                'y2': float(box.xyxy[0][3]),
                'class': int(box.cls[0])
            })
            
    print(f"\nDetections:\n{len(valid_detections)} STICKER detections")
    
    for i, det in enumerate(valid_detections, 1):
        print(f"\nDetection {i}:")
        print(f"- confidence: {det['conf']:.4f}")
        print(f"- x1: {det['x1']:.1f}")
        print(f"- y1: {det['y1']:.1f}")
        print(f"- x2: {det['x2']:.1f}")
        print(f"- y2: {det['y2']:.1f}")
        
    print("\nFinal:")
    if len(valid_detections) > 0:
        print("PASS")
    else:
        print("FAIL")

def main():
    parser = argparse.ArgumentParser(description="Test YOLO Detection Model on a single image.")
    parser.add_argument("image_path", help="Path to the image to test")
    parser.add_argument("--conf", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    args = parser.parse_args()
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runs', 'detect', 'fan_sticker_detection', 'weights', 'best.pt'))
    test_image(model_path, args.image_path, args.conf)

if __name__ == "__main__":
    main()
