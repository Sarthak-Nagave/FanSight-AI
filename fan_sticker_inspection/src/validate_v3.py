import os
import cv2
import hashlib
from collections import defaultdict
from PIL import Image

def get_image_hash(filepath):
    """Return MD5 hash of a file for exact duplicate detection."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except:
        return None

def validate_dataset(dataset_dir):
    print(f"--- VALIDATING DATASET: {dataset_dir} ---")
    
    classes = ["STICKER_PRESENT", "STICKER_MISSING"]
    total_images = 0
    corrupted_images = []
    
    # Store hashes to find exact duplicates
    # hash -> list of file paths
    file_hashes = defaultdict(list)
    
    # Track dimensions
    dimension_counts = defaultdict(int)
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Directory missing - {cls_dir}")
            continue
            
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(files)
        total_images += count
        
        print(f"[{cls}]: {count} images found.")
        
        for f in files:
            filepath = os.path.join(cls_dir, f)
            
            # Check for corruption / openable
            try:
                # Using PIL first to ensure it's a valid image file
                with Image.open(filepath) as img:
                    img.verify()
                
                # Check dimensions with cv2 as it's used in ML
                cv_img = cv2.imread(filepath)
                if cv_img is None:
                    raise ValueError("cv2 could not read image.")
                
                h, w = cv_img.shape[:2]
                dimension_counts[f"{w}x{h}"] += 1
                
                # Hash for duplicates
                file_hash = get_image_hash(filepath)
                if file_hash:
                    file_hashes[file_hash].append(filepath)
                    
            except Exception as e:
                corrupted_images.append(filepath)
    
    print("\n--- RESULTS ---")
    print(f"Total Valid Images: {total_images - len(corrupted_images)}")
    
    if corrupted_images:
        print(f"\n[!] CORRUPTED IMAGES FOUND ({len(corrupted_images)}):")
        for c in corrupted_images:
            print(f"  - {c}")
    else:
        print("\n[+] No corrupted images detected.")
        
    # Duplicates check
    duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
    if duplicates:
        print(f"\n[!] EXACT DUPLICATES FOUND ({len(duplicates)} sets):")
        for h, paths in duplicates.items():
            print(f"  - Hash {h[:8]}... used in {len(paths)} files:")
            for p in paths:
                print(f"      {p}")
    else:
        print("\n[+] No exact duplicates detected.")
        
    print("\n--- IMAGE DIMENSIONS ---")
    for dim, count in dimension_counts.items():
        print(f"  - {dim}: {count} images")
        
    print("\nValidation complete. Ensure background, lighting, and orientation variation visually!")

if __name__ == "__main__":
    # Base directory for V3 dataset
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Fan_Sticker_Dataset_V3"))
    validate_dataset(dataset_path)
