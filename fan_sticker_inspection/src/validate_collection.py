import os
import glob
import math
from PIL import Image, ImageDraw

def create_contact_sheet(image_paths, output_path, cols=10, thumb_size=150):
    if not image_paths:
        return False
        
    rows = math.ceil(len(image_paths) / cols)
    sheet_width = cols * thumb_size
    sheet_height = rows * thumb_size
    
    sheet = Image.new('RGB', (sheet_width, sheet_height), color='white')
    
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            img = img.convert('RGB')
            w, h = img.size
            
            # Crop to square
            if w > h:
                left = (w - h) // 2
                img = img.crop((left, 0, left + h, h))
            elif h > w:
                top = (h - w) // 2
                img = img.crop((0, top, w, top + w))
                
            img = img.resize((thumb_size, thumb_size))
            
            draw = ImageDraw.Draw(img)
            name = os.path.basename(path)
            draw.text((5, 5), name, fill='red')
            
            row = i // cols
            col = i % cols
            sheet.paste(img, (col * thumb_size, row * thumb_size))
        except Exception as e:
            print(f"Warning: Could not process {path} - {e}")
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, quality=85)
    return True

def main():
    dataset_dir = r"C:\Users\Lenovo\Desktop\Project\Fan_Sticker\Fan_Sticker_Dataset_V2"
    base_dir = r"C:\Users\Lenovo\Desktop\Project\Fan_Sticker\fan_sticker_inspection"
    
    present_dir = os.path.join(dataset_dir, "STICKER_PRESENT")
    missing_dir = os.path.join(dataset_dir, "STICKER_MISSING")
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def get_images(d):
        if not os.path.exists(d): return []
        imgs = glob.glob(os.path.join(d, "*.*"))
        return [p for p in imgs if os.path.splitext(p)[1].lower() in valid_exts]

    present_imgs = get_images(present_dir)
    missing_imgs = get_images(missing_dir)
    
    total_present = len(present_imgs)
    total_missing = len(missing_imgs)
    total = total_present + total_missing
    
    print("=" * 50)
    print("DATASET V2 COLLECTION VALIDATOR")
    print("=" * 50)
    print(f"Total Images: {total}")
    print(f" - STICKER_PRESENT: {total_present}")
    print(f" - STICKER_MISSING: {total_missing}")
    
    if total == 0:
        print("\nStatus: EMPTY. No images found in Fan_Sticker_Dataset_V2.")
        return
        
    print("\n[Class Balance]")
    if total > 0:
        p_ratio = total_present / total * 100
        m_ratio = total_missing / total * 100
        print(f"PRESENT: {p_ratio:.1f}% | MISSING: {m_ratio:.1f}%")
        if abs(p_ratio - 50) > 10:
            print("⚠️ WARNING: Dataset is imbalanced. Aim for a 50/50 split.")
        else:
            print("✅ Dataset is well balanced.")
            
    if total_present < 100 or total_missing < 100:
        print("⚠️ WARNING: You have fewer than 100 images per class. More images are recommended.")
        
    print("\n[Orientation Distribution]")
    print("Automated precise orientation extraction is difficult without a trained model.")
    print("Generating visual contact sheets for manual verification...")
    
    out_dir = os.path.join(base_dir, "review")
    p_sheet = os.path.join(out_dir, "v2_present_orientations.jpg")
    m_sheet = os.path.join(out_dir, "v2_missing_orientations.jpg")
    
    if create_contact_sheet(present_imgs, p_sheet):
        print(f"✅ Created {p_sheet}")
    if create_contact_sheet(missing_imgs, m_sheet):
        print(f"✅ Created {m_sheet}")
        
    print("\nACTION REQUIRED:")
    print("Please open the generated contact sheets in the 'review' folder.")
    print("Visually verify that BOTH classes show fans in all 360-degree rotations.")
    print("If one class predominantly points in one direction, your dataset still has orientation bias.")
    print("=" * 50)

if __name__ == "__main__":
    main()
