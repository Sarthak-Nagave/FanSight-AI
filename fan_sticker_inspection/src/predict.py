import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config, get_prediction_result

def predict_image(image_path):
    model_path = Path('runs/classify/fan_sticker_v2/weights/best.pt')
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return
        
    # Default threshold
    threshold = 0.70
    try:
        config = load_config()
        if 'inference' in config and 'confidence_threshold' in config['inference']:
            threshold = float(config['inference']['confidence_threshold'])
    except:
        pass
    
    model = YOLO(str(model_path))
    results = model(image_path, verbose=False)
    
    # Process results
    result = results[0]
    probs = result.probs
    top1_index = probs.top1
    top1_conf = float(probs.top1conf)
    top1_class_name = result.names[top1_index]
    
    final_result = get_prediction_result(top1_class_name, top1_conf, threshold)
    
    print(f"Prediction: {top1_class_name}")
    print(f"Confidence: {top1_conf:.1%}")
    print(f"Result: {final_result}")
    
    return top1_class_name, top1_conf, final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fan sticker presence")
    parser.add_argument("image", help="Path to image file")
    args = parser.parse_args()
    
    predict_image(args.image)
