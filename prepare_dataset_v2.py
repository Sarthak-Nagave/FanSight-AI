import os
import shutil
import random
import hashlib
from pathlib import Path
from PIL import Image

def get_image_hash(filepath):
    """Generate MD5 hash for a file to check for exact duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_corrupted(filepath):
    """Check if an image can be opened and verified by PIL."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return False
    except Exception:
        return True

# Paths
source_dir = Path("Fan_Project_Dataset")
dest_dir = Path("fan_sticker_inspection/dataset")

# Classes mappings
classes_map = {
    "With_Sticker_Imgs": "STICKER_PRESENT",
    "No_Sticker_Imgs": "STICKER_MISSING"
}

# Clean existing dataset
if dest_dir.exists():
    print(f"Cleaning existing dataset at {dest_dir}...")
    shutil.rmtree(dest_dir)

# Create output directories
for split in ['train', 'val']:
    for new_cls in classes_map.values():
        (dest_dir / split / new_cls).mkdir(parents=True, exist_ok=True)

# Split ratio
val_ratio = 0.2

stats = {"total": 0, "train": 0, "val": 0, "classes": {}}
corrupted_files = []
duplicates = []
global_hashes = {} # hash -> filepath

for orig_cls, new_cls in classes_map.items():
    cls_dir = source_dir / orig_cls
    if not cls_dir.exists():
        print(f"Warning: {cls_dir} does not exist.")
        continue
        
    print(f"Processing {orig_cls} -> {new_cls}...")
    all_images = list(cls_dir.glob("*.jpg"))
    valid_images = []
    
    for img_path in all_images:
        if is_corrupted(img_path):
            corrupted_files.append(str(img_path))
            continue
            
        img_hash = get_image_hash(img_path)
        if img_hash in global_hashes:
            duplicates.append((str(img_path), str(global_hashes[img_hash])))
            continue
            
        global_hashes[img_hash] = img_path
        valid_images.append(img_path)
        
    random.shuffle(valid_images)
    
    val_count = int(len(valid_images) * val_ratio)
    val_images = valid_images[:val_count]
    train_images = valid_images[val_count:]
    
    stats["classes"][new_cls] = {"total": len(valid_images), "train": len(train_images), "val": len(val_images)}
    stats["total"] += len(valid_images)
    stats["train"] += len(train_images)
    stats["val"] += len(val_images)
    
    for img in train_images:
        shutil.copy(img, dest_dir / "train" / new_cls / img.name)
        
    for img in val_images:
        shutil.copy(img, dest_dir / "val" / new_cls / img.name)

print("\n--- Dataset Preparation Report ---")
print(f"Total STICKER_PRESENT: {stats['classes'].get('STICKER_PRESENT', {}).get('total', 0)}")
print(f"Total STICKER_MISSING: {stats['classes'].get('STICKER_MISSING', {}).get('total', 0)}")
print(f"Train STICKER_PRESENT: {stats['classes'].get('STICKER_PRESENT', {}).get('train', 0)}")
print(f"Train STICKER_MISSING: {stats['classes'].get('STICKER_MISSING', {}).get('train', 0)}")
print(f"Validation STICKER_PRESENT: {stats['classes'].get('STICKER_PRESENT', {}).get('val', 0)}")
print(f"Validation STICKER_MISSING: {stats['classes'].get('STICKER_MISSING', {}).get('val', 0)}")
print(f"Total train images: {stats['train']}")
print(f"Total validation images: {stats['val']}")

print(f"\nCorrupted Images Found: {len(corrupted_files)}")
if corrupted_files:
    for f in corrupted_files:
        print(f" - {f}")

print(f"Duplicate/Suspicious Images Found (ignored): {len(duplicates)}")
if duplicates:
    for f1, f2 in duplicates:
        print(f" - {f1} is a duplicate of {f2}")
