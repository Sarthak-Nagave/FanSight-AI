import os
import glob
import math
from PIL import Image, ImageDraw, ImageFont

def make_contact_sheet(image_paths, output_path, cols=10, thumb_size=150):
    if not image_paths:
        print(f"No images for {output_path}")
        return
        
    rows = math.ceil(len(image_paths) / cols)
    sheet_width = cols * thumb_size
    sheet_height = rows * thumb_size
    
    sheet = Image.new('RGB', (sheet_width, sheet_height), color='white')
    
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            img = img.convert('RGB')
            # Resize and crop to square
            w, h = img.size
            if w > h:
                left = (w - h) // 2
                img = img.crop((left, 0, left + h, h))
            elif h > w:
                top = (h - w) // 2
                img = img.crop((0, top, w, top + w))
                
            img = img.resize((thumb_size, thumb_size))
            
            # Draw filename
            draw = ImageDraw.Draw(img)
            name = os.path.basename(path)
            draw.text((5, 5), name, fill='red')
            
            row = i // cols
            col = i % cols
            sheet.paste(img, (col * thumb_size, row * thumb_size))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, quality=85)
    print(f"Saved {output_path} with {len(image_paths)} images.")

def main():
    base_dir = r"C:\Users\Lenovo\Desktop\Project\Fan_Sticker"
    present_dir = os.path.join(base_dir, "Fan_Project_Dataset", "With_Sticker_Imgs")
    missing_dir = os.path.join(base_dir, "Fan_Project_Dataset", "No_Sticker_Imgs")
    
    present_imgs = glob.glob(os.path.join(present_dir, "*.*"))
    missing_imgs = glob.glob(os.path.join(missing_dir, "*.*"))
    
    # Filter valid extensions
    valid_exts = {'.jpg', '.jpeg', '.png'}
    present_imgs = [p for p in present_imgs if os.path.splitext(p)[1].lower() in valid_exts]
    missing_imgs = [p for p in missing_imgs if os.path.splitext(p)[1].lower() in valid_exts]
    
    out_dir = os.path.join(base_dir, "fan_sticker_inspection", "review")
    make_contact_sheet(present_imgs, os.path.join(out_dir, "present_orientations.jpg"))
    make_contact_sheet(missing_imgs, os.path.join(out_dir, "missing_orientations.jpg"))

if __name__ == "__main__":
    main()
