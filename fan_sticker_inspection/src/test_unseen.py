import os
from pathlib import Path
from ultralytics import YOLO

def test_unseen():
    model_path = 'runs/classify/runs/classify/fan_sticker/weights/best.pt'
    if not os.path.exists(model_path):
        # Fallback to standard path if it got moved
        model_path = 'runs/classify/fan_sticker/weights/best.pt'
        
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    test_dir = Path('test_unseen')
    
    classes = ['STICKER_PRESENT', 'STICKER_MISSING']
    
    total_images = 0
    correct_predictions = 0
    incorrect_images = []
    
    class_stats = {cls: {'total': 0, 'correct': 0, 'confidences': []} for cls in classes}
    confusion_matrix = {expected: {predicted: 0 for predicted in classes} for expected in classes}
    
    for expected_class in classes:
        class_dir = test_dir / expected_class
        if not class_dir.exists():
            continue
            
        for img_path in class_dir.glob('*.*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
                
            total_images += 1
            class_stats[expected_class]['total'] += 1
            
            # Predict
            results = model(img_path, verbose=False)
            result = results[0]
            
            # Get predicted class and confidence
            pred_idx = result.probs.top1
            pred_class = result.names[pred_idx]
            conf = result.probs.top1conf.item()
            
            class_stats[expected_class]['confidences'].append(conf)
            confusion_matrix[expected_class][pred_class] += 1
            
            is_correct = pred_class == expected_class
            if is_correct:
                correct_predictions += 1
                class_stats[expected_class]['correct'] += 1
                status = "CORRECT"
            else:
                status = "FAIL"
                incorrect_images.append({
                    'filename': img_path.name,
                    'expected': expected_class,
                    'predicted': pred_class,
                    'confidence': conf
                })
            
            print(f"\n{img_path.name}")
            print(f"Expected: {expected_class}")
            print(f"Predicted: {pred_class}")
            print(f"Confidence: {conf * 100:.1f}%")
            print(f"Result: {status}")

    print("\n" + "="*40)
    print("TASK 4 — OVERALL TEST REPORT")
    print("="*40)
    
    if total_images == 0:
        print("No independent unseen images are available yet.")
        return
        
    accuracy = correct_predictions / total_images * 100
    print(f"Total unseen images: {total_images}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_images - correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%\n")
    
    for cls in classes:
        stats = class_stats[cls]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{cls} accuracy: {acc:.2f}% ({stats['correct']}/{stats['total']})")
        
    print("\nConfusion Matrix (Expected \\ Predicted):")
    header = f"{'':<18}" + "".join([f"{c:<18}" for c in classes])
    print(header)
    for expected in classes:
        row = f"{expected:<18}"
        for predicted in classes:
            row += f"{confusion_matrix[expected][predicted]:<18}"
        print(row)
        
    if incorrect_images:
        print("\nIncorrectly classified images:")
        for inc in incorrect_images:
            print(f" - {inc['filename']} (Expected: {inc['expected']}, Predicted: {inc['predicted']} with conf {inc['confidence']:.1%})")
    else:
        print("\nIncorrectly classified images: None")

    print("\n" + "="*40)
    print("TASK 5 — CONFIDENCE ANALYSIS")
    print("="*40)
    
    for cls in classes:
        confs = class_stats[cls]['confidences']
        if confs:
            min_c = min(confs) * 100
            max_c = max(confs) * 100
            avg_c = (sum(confs) / len(confs)) * 100
            print(f"{cls}:")
            print(f" - Minimum confidence: {min_c:.1f}%")
            print(f" - Maximum confidence: {max_c:.1f}%")
            print(f" - Average confidence: {avg_c:.1f}%\n")
        else:
            print(f"{cls}: No images tested.\n")

if __name__ == '__main__':
    test_unseen()
