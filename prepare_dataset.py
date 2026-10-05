import os
import shutil
import random
from pathlib import Path

# Paths
source_dir = Path("Fan_Project_Dataset")
dest_dir = Path("fan_sticker_inspection/dataset")

# Classes mappings
classes_map = {
    "With_Sticker_Imgs": "STICKER_PRESENT",
    "No_Sticker_Imgs": "STICKER_MISSING"
}

# Create output directories
for split in ['train', 'val']:
    for new_cls in classes_map.values():
        (dest_dir / split / new_cls).mkdir(parents=True, exist_ok=True)

# Split ratio
val_ratio = 0.2

stats = {"total": 0, "train": 0, "val": 0, "classes": {}}

for orig_cls, new_cls in classes_map.items():
    cls_dir = source_dir / orig_cls
    if not cls_dir.exists():
        print(f"Warning: {cls_dir} does not exist.")
        continue
        
    images = list(cls_dir.glob("*.jpg"))
    random.shuffle(images)
    
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]
    train_images = images[val_count:]
    
    stats["classes"][new_cls] = {"total": len(images), "train": len(train_images), "val": len(val_images)}
    stats["total"] += len(images)
    stats["train"] += len(train_images)
    stats["val"] += len(val_images)
    
    for img in train_images:
        shutil.copy(img, dest_dir / "train" / new_cls / img.name)
        
    for img in val_images:
        shutil.copy(img, dest_dir / "val" / new_cls / img.name)

print("Dataset Preparation Completed.")
print(f"Total Images: {stats['total']}")
print(f"Training Images: {stats['train']}")
print(f"Validation Images: {stats['val']}")
for cls, cls_stats in stats['classes'].items():
    print(f"Class {cls}: {cls_stats['total']} total (Train: {cls_stats['train']}, Val: {cls_stats['val']})")
