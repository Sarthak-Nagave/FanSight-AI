import os
import shutil
import random
import glob
from collections import defaultdict

def group_by_sequence(files):
    groups = defaultdict(list)
    for f in files:
        basename = os.path.basename(f[0])
        prefix = basename[:10] if len(basename) > 10 else basename
        groups[prefix].append(f)
    return list(groups.values())

def balance_split(groups, target_ratio=0.8):
    # Sort groups by size descending to pack them nicely (greedy approach)
    groups.sort(key=len, reverse=True)
    
    total_imgs = sum(len(g) for g in groups)
    target_train_count = int(total_imgs * target_ratio)
    
    train_groups = []
    val_groups = []
    
    current_train_count = 0
    for g in groups:
        if current_train_count + len(g) <= target_train_count or len(train_groups) == 0:
            train_groups.append(g)
            current_train_count += len(g)
        else:
            val_groups.append(g)
            
    # Flatten
    train_files = [f for g in train_groups for f in g]
    val_files = [f for g in val_groups for f in g]
    return train_files, val_files

def main():
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = os.path.join(proj_dir, "dataset_detection")
    
    # We will just rearrange the images and labels between train/val in the existing dataset_detection
    all_present_imgs = glob.glob(os.path.join(dataset_dir, "images", "*", "*.jpg")) + glob.glob(os.path.join(dataset_dir, "images", "*", "*.png"))
    
    # To identify present/missing, we look at the labels
    present_files = []
    missing_files = []
    
    for img_path in all_present_imgs:
        name = os.path.basename(img_path)
        split = "train" if "images\\train" in img_path or "images/train" in img_path else "val"
        lbl_path = os.path.join(dataset_dir, "labels", split, os.path.splitext(name)[0] + ".txt")
        
        if os.path.getsize(lbl_path) > 0:
            present_files.append((img_path, lbl_path))
        else:
            missing_files.append((img_path, lbl_path))
            
    # Group them
    p_groups = group_by_sequence(present_files)
    m_groups = group_by_sequence(missing_files)
    
    random.seed(42)
    random.shuffle(p_groups)
    random.shuffle(m_groups)
    
    # Split to hit 80/20 of images, not groups
    p_train, p_val = balance_split(p_groups, 0.8)
    m_train, m_val = balance_split(m_groups, 0.8)
    
    print(f"Targeting 80/20 split:")
    print(f"PRESENT: Train {len(p_train)} ({(len(p_train)/len(present_files))*100:.1f}%), Val {len(p_val)}")
    print(f"MISSING: Train {len(m_train)} ({(len(m_train)/len(missing_files))*100:.1f}%), Val {len(m_val)}")
    
    # Move files to temp to avoid collisions, then to final
    temp_dir = os.path.join(dataset_dir, "temp_reorg")
    os.makedirs(temp_dir, exist_ok=True)
    
    for img, lbl in (p_train + m_train + p_val + m_val):
        shutil.move(img, os.path.join(temp_dir, os.path.basename(img)))
        shutil.move(lbl, os.path.join(temp_dir, os.path.basename(lbl)))
        
    def move_to_split(files, split_name):
        for img, lbl in files:
            name = os.path.basename(img)
            lbl_name = os.path.basename(lbl)
            shutil.move(os.path.join(temp_dir, name), os.path.join(dataset_dir, "images", split_name, name))
            shutil.move(os.path.join(temp_dir, lbl_name), os.path.join(dataset_dir, "labels", split_name, lbl_name))
            
    move_to_split(p_train + m_train, "train")
    move_to_split(p_val + m_val, "val")
    
    shutil.rmtree(temp_dir)
    print("Reorganization complete.")

if __name__ == "__main__":
    main()
