"""Quick test: verify sticker locator works on real images."""
import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.sticker_locator import locate_sticker, get_expected_sticker_region

base = r"C:\Users\Lenovo\Desktop\Project\Fan_Sticker\Fan_Project_Dataset"

# Test on PRESENT images
print("=== STICKER PRESENT ===")
present_dir = os.path.join(base, "With_Sticker_Imgs")
files = sorted(os.listdir(present_dir))[:5]
for f in files:
    img = cv2.imread(os.path.join(present_dir, f))
    h, w = img.shape[:2]
    result = locate_sticker(img)
    if result:
        print(f"  {f}: FOUND at ({result['cx']}, {result['cy']}) r={result['radius']}  img={w}x{h}")
    else:
        print(f"  {f}: NOT FOUND  img={w}x{h}")

# Test on MISSING images
print("\n=== STICKER MISSING ===")
missing_dir = os.path.join(base, "No_Sticker_Imgs")
files = sorted(os.listdir(missing_dir))[:5]
for f in files:
    img = cv2.imread(os.path.join(missing_dir, f))
    h, w = img.shape[:2]
    result_found = locate_sticker(img)
    result_expected = get_expected_sticker_region(img)
    found_str = f"FOUND at ({result_found['cx']}, {result_found['cy']}) r={result_found['radius']}" if result_found else "NOT FOUND (correct)"
    print(f"  {f}: {found_str}")
    print(f"         Expected area: ({result_expected['cx']}, {result_expected['cy']}) r={result_expected['radius']}")
