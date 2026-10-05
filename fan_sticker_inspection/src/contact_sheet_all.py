import os
import cv2
import numpy as np
import math

def create_contact_sheet(image_paths, output_path, cols=6, thumb_size=200):
    if not image_paths:
        print("No images found.")
        return

    n_images = len(image_paths)
    rows = math.ceil(n_images / cols)
    
    canvas = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
    
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
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
        
    cv2.imwrite(output_path, canvas)
    print(f"Generated {output_path} with {n_images} images.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Fan_Project_Dataset"))
    present_dir = os.path.join(base_dir, "With_Sticker_Imgs")
    missing_dir = os.path.join(base_dir, "No_Sticker_Imgs")
    
    if os.path.exists(present_dir):
        files = [os.path.join(present_dir, f) for f in os.listdir(present_dir) if f.lower().endswith(('.jpg', '.png'))]
        create_contact_sheet(files, os.path.join(base_dir, "contact_sheet_PRESENT_ALL.jpg"))
        
    if os.path.exists(missing_dir):
        files = [os.path.join(missing_dir, f) for f in os.listdir(missing_dir) if f.lower().endswith(('.jpg', '.png'))]
        create_contact_sheet(files, os.path.join(base_dir, "contact_sheet_MISSING_ALL.jpg"))
