import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config

def validate_model():
    model_path = Path('runs/train/fan_sticker_cls/weights/best.pt')
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Train the model first.")
        return
        
    config = load_config()
    dataset_path = Path(config['dataset']['path']).absolute()
    
    model = YOLO(str(model_path))
    
    print("Running validation...")
    metrics = model.val(
        data=str(dataset_path),
        project='runs/val',
        name='fan_sticker_val'
    )
    
    print("\n--- Validation Results ---")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    
if __name__ == "__main__":
    validate_model()
