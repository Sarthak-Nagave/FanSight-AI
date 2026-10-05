import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config, get_prediction_result

def run_webcam(camera_id=0):
    model_path = Path('runs/train/fan_sticker_cls/weights/best.pt')
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return
        
    config = load_config()
    threshold = config['inference']['confidence_threshold']
    
    model = YOLO(str(model_path))
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        results = model(frame, verbose=False)
        result = results[0]
        probs = result.probs
        
        if probs is not None:
            top1_index = probs.top1
            top1_conf = float(probs.top1conf)
            top1_class_name = result.names[top1_index]
            
            final_result = get_prediction_result(top1_class_name, top1_conf, threshold)
            
            # Display information
            color = (0, 255, 0) if final_result == "PASS" else (0, 0, 255) if final_result == "FAIL" else (0, 255, 255)
            
            text = f"{final_result} | {top1_class_name} ({top1_conf:.1%})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        cv2.imshow("Fan Sticker Inspection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
