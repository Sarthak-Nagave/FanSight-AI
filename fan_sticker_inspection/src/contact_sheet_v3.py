import os
import cv2
import numpy as np
import math

def create_contact_sheet(image_paths, output_path, cols=5, thumb_size=200):
    """Creates a grid of images for visual inspection."""
    if not image_paths:
        print("No images provided for contact sheet.")
        return

    n_images = len(image_paths)
    rows = math.ceil(n_images / cols)
    
    # Create empty canvas (black background)
    canvas = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
    
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Resize maintaining aspect ratio with padding
        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # Center in thumbnail box
        thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
        y_offset = (thumb_size - new_h) // 2
        x_offset = (thumb_size - new_w) // 2
        thumb[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Calculate grid position
        row = idx // cols
        col = idx % cols
        y_start = row * thumb_size
        x_start = col * thumb_size
        
        canvas[y_start:y_start+thumb_size, x_start:x_start+thumb_size] = thumb
        
    cv2.imwrite(output_path, canvas)
    print(f"Generated contact sheet: {output_path}")

def main():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Fan_Sticker_Dataset_V3"))
    classes = ["STICKER_PRESENT", "STICKER_MISSING"]
    
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_dir):
            continue
            
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if files:
            # Sort to ensure consistent order, maybe take top 25 for a 5x5 sheet
            files.sort()
            sample_files = files[:25] # Limit to 25 for the sheet
            
            output_file = os.path.join(dataset_path, f"contact_sheet_{cls}.jpg")
            create_contact_sheet(sample_files, output_file, cols=5, thumb_size=200)

if __name__ == "__main__":
    main()
