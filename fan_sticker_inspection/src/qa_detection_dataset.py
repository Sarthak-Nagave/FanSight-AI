import os
import cv2
import numpy as np
import glob
import math
from collections import defaultdict
import hashlib

def get_hash(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def check_dataset_split(dataset_dir):
    print("--- 1. DATASET SPLIT ANALYSIS ---")
    splits = ['train', 'val']
    
    total_imgs = 0
    img_hashes = {'train': set(), 'val': set()}
    leakage_count = 0
    
    stats = {'train': {'present': 0, 'missing': 0}, 'val': {'present': 0, 'missing': 0}}
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, 'images', split)
        lbl_dir = os.path.join(dataset_dir, 'labels', split)
        
        if not os.path.exists(img_dir):
            continue
            
        imgs = glob.glob(os.path.join(img_dir, "*.[jJ][pP]*[gG]"))
        total_imgs += len(imgs)
        
        for img_path in imgs:
            name = os.path.basename(img_path)
            txt_path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            
            # Determine if it's PRESENT or MISSING based on label file contents
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                stats[split]['present'] += 1
            else:
                stats[split]['missing'] += 1
                
            img_hash = get_hash(img_path)
            img_hashes[split].add(img_hash)
            
    print(f"TRAIN: {stats['train']['present']} PRESENT, {stats['train']['missing']} MISSING")
    print(f"VAL:   {stats['val']['present']} PRESENT, {stats['val']['missing']} MISSING")
    
    # Check Leakage
    leaked = img_hashes['train'].intersection(img_hashes['val'])
    print(f"\nExact duplicates across train/val: {len(leaked)}")
    
    # Analyze the grouping distribution
    # If the distribution is skewed, it's because splitting by group count doesn't split by image count evenly.
    
    return stats

def verify_labels_and_render(dataset_dir):
    print("\n--- 2. LABEL VERIFICATION & RENDERING ---")
    review_dir = os.path.join(dataset_dir, "review")
    os.makedirs(review_dir, exist_ok=True)
    
    label_stats = {
        'PRESENT': {'total': 0, 'valid': 0, 'missing_file': 0, 'invalid_coord': 0, 'multi_label': 0},
        'MISSING': {'total': 0, 'empty': 0, 'non_empty': 0}
    }
    
    # For rendering contact sheets
    rendered_images = defaultdict(list)
    
    splits = ['train', 'val']
    for split in splits:
        img_dir = os.path.join(dataset_dir, 'images', split)
        lbl_dir = os.path.join(dataset_dir, 'labels', split)
        
        if not os.path.exists(img_dir):
            continue
            
        imgs = glob.glob(os.path.join(img_dir, "*.[jJ][pP]*[gG]"))
        imgs.sort()
        
        for img_path in imgs:
            name = os.path.basename(img_path)
            txt_path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            
            is_present = os.path.exists(txt_path) and os.path.getsize(txt_path) > 0
            
            if not os.path.exists(txt_path):
                if is_present: label_stats['PRESENT']['missing_file'] += 1
                else: label_stats['MISSING']['missing_file'] += 1 # technically not tracked
                continue
                
            # Read label
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            lines = [l.strip() for l in lines if l.strip()]
            
            cv_img = cv2.imread(img_path)
            h, w = cv_img.shape[:2]
            
            if len(lines) == 0:
                # Expecting MISSING
                label_stats['MISSING']['total'] += 1
                label_stats['MISSING']['empty'] += 1
                
                # Render for review sheet
                # Just scale down
                rendered_images[f'{split}_missing'].append(cv_img)
            else:
                label_stats['PRESENT']['total'] += 1
                if len(lines) > 1:
                    label_stats['PRESENT']['multi_label'] += 1
                    
                valid = True
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        valid = False
                        break
                    c_id, nx, ny, nw, nh = map(float, parts)
                    if c_id != 0 or not (0 <= nx <= 1) or not (0 <= ny <= 1) or nw <= 0 or nh <= 0:
                        valid = False
                        break
                        
                    # Draw box
                    bx = int((nx - nw/2) * w)
                    by = int((ny - nh/2) * h)
                    bw = int(nw * w)
                    bh = int(nh * h)
                    
                    cv2.rectangle(cv_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 12)
                    
                if valid:
                    label_stats['PRESENT']['valid'] += 1
                else:
                    label_stats['PRESENT']['invalid_coord'] += 1
                    
                rendered_images[f'{split}_present'].append(cv_img)
                
    print("\nPRESENT images:")
    for k, v in label_stats['PRESENT'].items():
        print(f"  {k}: {v}")
        
    print("\nMISSING images:")
    for k, v in label_stats['MISSING'].items():
        print(f"  {k}: {v}")
        
    print("\n--- 3. GENERATING CONTACT SHEETS ---")
    
    for category, img_list in rendered_images.items():
        if not img_list: continue
        
        # Limit to 30 images max per sheet for review
        img_list = img_list[:30]
        
        thumb_size = 250
        cols = 5
        rows = math.ceil(len(img_list) / cols)
        
        canvas = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
        
        for idx, img in enumerate(img_list):
            ih, iw = img.shape[:2]
            scale = thumb_size / max(ih, iw)
            new_w, new_h = int(iw * scale), int(ih * scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
            y_offset = (thumb_size - new_h) // 2
            x_offset = (thumb_size - new_w) // 2
            thumb[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            row = idx // cols
            col = idx % cols
            y_start = row * thumb_size
            x_start = col * thumb_size
            
            canvas[y_start:y_start+thumb_size, x_start:x_start+thumb_size] = thumb
            
        out_path = os.path.join(review_dir, f"review_{category}.jpg")
        cv2.imwrite(out_path, canvas)
        print(f"Saved {out_path} ({len(img_list)} images rendered)")

def check_yaml(dataset_dir):
    print("\n--- 4. CHECK YAML ---")
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            print(f.read())
    else:
        print("[!] dataset.yaml missing")

if __name__ == "__main__":
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset_detection"))
    check_dataset_split(dataset_dir)
    verify_labels_and_render(dataset_dir)
    check_yaml(dataset_dir)
