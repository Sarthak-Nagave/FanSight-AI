import os
import torch
from ultralytics import YOLO
import time

def main():
    print("--- YOLO11n DETECTION TRAINING ---")
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_detection', 'dataset.yaml'))
    
    print(f"Dataset YAML: {yaml_path}")
    
    model = YOLO('yolo11n.pt')
    
    start_time = time.time()
    
    try:
        print("\nStarting training with batch=8...")
        results = model.train(
            data=yaml_path,
            epochs=50,
            imgsz=640,
            batch=8,
            workers=0,
            project='runs/detect',
            name='fan_sticker_detection',
            device=device,
            exist_ok=True # Overwrite if exists to ensure clean run
        )
    except Exception as e:
        if "CUDA out of memory" in str(e):
            print("\n[!] CUDA Out of Memory with batch=8. Retrying with batch=4...")
            # Empty cache and retry
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            model = YOLO('yolo11n.pt')
            results = model.train(
                data=yaml_path,
                epochs=50,
                imgsz=640,
                batch=4,
                workers=0,
                project='runs/detect',
                name='fan_sticker_detection',
                device=device,
                exist_ok=True
            )
        else:
            raise e
            
    end_time = time.time()
    train_duration = (end_time - start_time) / 60.0
    
    print("\n--- TRAINING COMPLETED ---")
    print(f"Total time: {train_duration:.2f} minutes")
    
    # Run validation on the best model to get precise metrics
    best_model_path = os.path.join('runs', 'detect', 'fan_sticker_detection', 'weights', 'best.pt')
    print(f"\nLoading best model from {best_model_path} for final validation...")
    
    best_model = YOLO(best_model_path)
    val_results = best_model.val(data=yaml_path, imgsz=640, device=device)
    
    metrics = val_results.results_dict
    
    print("\n--- FINAL METRICS ---")
    print(f"Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall:    {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP50:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP50-95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
