import os
import cv2
import shutil
import random
import glob
from collections import defaultdict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sticker_locator import locate_sticker

def group_by_sequence(files):
    # Filenames are like 1791104117514.jpg
    # Group by the first 10 characters (seconds timestamp)
    groups = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        prefix = basename[:10] if len(basename) > 10 else basename
        groups[prefix].append(f)
    return groups

def setup_directories(base_path):
    dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

def process_and_annotate(src_files, cls_id, is_present, dest_img_dir, dest_lbl_dir):
    processed = 0
    failed = 0
    
    for src in src_files:
        filename = os.path.basename(src)
        name, _ = os.path.splitext(filename)
        
        img = cv2.imread(src)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Copy image
        dst_img = os.path.join(dest_img_dir, filename)
        shutil.copy(src, dst_img)
        
        # Create label
        dst_lbl = os.path.join(dest_lbl_dir, f"{name}.txt")
        
        if is_present:
            res = locate_sticker(img)
            if res:
                cx, cy, r = res['cx'], res['cy'], res['radius']
                # Normalize
                nx = cx / w
                ny = cy / h
                nw = (r * 2) / w
                nh = (r * 2) / h
                
                # Constrain to 0-1
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                
                with open(dst_lbl, 'w') as f:
                    f.write(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
                processed += 1
            else:
                print(f"[!] Auto-annotation failed for: {filename}")
                failed += 1
                # If it fails to find sticker in a PRESENT image, we still need a label...
                # We can fallback to user annotation or just skip it.
                # Since we want to annotate, let's skip copying if we fail to find it automatically.
                os.remove(dst_img)
        else:
            # Missing sticker -> Empty label file
            open(dst_lbl, 'w').close()
            processed += 1
            
    return processed, failed

def main():
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(proj_dir, "Fan_Project_Dataset")
    dest_dir = os.path.join(proj_dir, "dataset_detection")
    
    present_dir = os.path.join(src_dir, "With_Sticker_Imgs")
    missing_dir = os.path.join(src_dir, "No_Sticker_Imgs")
    
    setup_directories(dest_dir)
    
    # Get files
    present_files = glob.glob(os.path.join(present_dir, "*.[jJ][pP]*[gG]"))
    missing_files = glob.glob(os.path.join(missing_dir, "*.[jJ][pP]*[gG]"))
    
    # Group by sequence to prevent train/val leakage
    present_groups = list(group_by_sequence(present_files).values())
    missing_groups = list(group_by_sequence(missing_files).values())
    
    random.seed(42)
    random.shuffle(present_groups)
    random.shuffle(missing_groups)
    
    # Split 80/20
    p_split = int(len(present_groups) * 0.8)
    m_split = int(len(missing_groups) * 0.8)
    
    present_train = [f for g in present_groups[:p_split] for f in g]
    present_val = [f for g in present_groups[p_split:] for f in g]
    
    missing_train = [f for g in missing_groups[:m_split] for f in g]
    missing_val = [f for g in missing_groups[m_split:] for f in g]
    
    print("--- STARTING AUTO-ANNOTATION ---")
    
    # Process train
    print("\nProcessing Train Set (PRESENT)...")
    pt, pt_f = process_and_annotate(present_train, 0, True, 
                                   os.path.join(dest_dir, "images/train"), 
                                   os.path.join(dest_dir, "labels/train"))
                                   
    print("\nProcessing Train Set (MISSING)...")
    mt, mt_f = process_and_annotate(missing_train, 0, False, 
                                   os.path.join(dest_dir, "images/train"), 
                                   os.path.join(dest_dir, "labels/train"))
                                   
    # Process val
    print("\nProcessing Val Set (PRESENT)...")
    pv, pv_f = process_and_annotate(present_val, 0, True, 
                                   os.path.join(dest_dir, "images/val"), 
                                   os.path.join(dest_dir, "labels/val"))
                                   
    print("\nProcessing Val Set (MISSING)...")
    mv, mv_f = process_and_annotate(missing_val, 0, False, 
                                   os.path.join(dest_dir, "images/val"), 
                                   os.path.join(dest_dir, "labels/val"))
                                   
    print("\n--- ANNOTATION SUMMARY ---")
    print(f"Train: {pt + mt} images (Present: {pt}, Missing: {mt})")
    print(f"Val:   {pv + mv} images (Present: {pv}, Missing: {mv})")
    if (pt_f + pv_f) > 0:
        print(f"Failed to auto-annotate {pt_f + pv_f} PRESENT images.")
        
    # Create dataset.yaml for YOLO
    yaml_path = os.path.join(dest_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {dest_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        f.write("  0: STICKER\n")
        
    print(f"\nCreated YOLO dataset config at: {yaml_path}")
    print("Ready for YOLO detection training.")

if __name__ == "__main__":
    main()
