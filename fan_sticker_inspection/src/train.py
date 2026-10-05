import os
from pathlib import Path
from ultralytics import YOLO
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config

def train_model():
    config = load_config()
    train_cfg = config['training']
    dataset_path = Path(config['dataset']['path']).absolute()
    
    # Initialize YOLO11n-cls model
    model = YOLO(train_cfg['model'])
    
    # Train the model
    print(f"Starting training with dataset at: {dataset_path}")
    results = model.train(
        data=str(dataset_path),
        epochs=train_cfg['epochs'],
        imgsz=train_cfg['imgsz'],
        batch=train_cfg['batch'],
        workers=train_cfg['workers'],
        device=train_cfg['device'] if train_cfg['device'] else None,
        project='runs/classify',
        name='fan_sticker'
    )
    
    print("Training complete!")
    print(f"Model saved to: runs/classify/fan_sticker/weights/best.pt")

if __name__ == "__main__":
    train_model()
