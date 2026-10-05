# Fan Sticker Inspection - Data Collection Guide (V2)

## 1. Purpose
The purpose of this V2 dataset is to eliminate the **orientation bias** found in the original dataset. The original dataset suffered from a severe bias where all `STICKER_PRESENT` images had the fan wire pointing down, and all `STICKER_MISSING` images had the wire pointing up. This caused the model to learn the wire direction instead of the sticker presence. 

This guide ensures we collect a robust dataset covering a full 360° range of fan rotations for **both** classes equally.

## 2. Camera Setup
- **Camera Height/Angle:** Keep the camera at a consistent overhead angle and distance from the fan.
- **Lighting:** Maintain consistent, evenly lit ambient lighting to avoid harsh shadows.
- **Background:** Use a consistent background (like the speckled countertop used previously).
- **Framing:** Keep the fan roughly centered, but allow for small position shifts (up/down/left/right) to improve model robustness.

## 3. Required Orientations
To ensure broad 360° coverage without bias, capture the fan at approximately these angles for **both** classes:
* 0° (Upright)
* 45°
* 90°
* 135°
* 180° (Upside down)
* 225°
* 270°
* 315°
* Random arbitrary angles between these markers.

Exact degree precision is not required. The goal is visual variety in rotation.

## 4. How to Capture STICKER_PRESENT Images
1. Take a fan **with** a sticker.
2. Place it at 0° (wire pointing down/right). Take 2-3 photos, shifting the fan slightly.
3. Rotate the fan ~45°. Take 2-3 photos.
4. Continue rotating and snapping photos until you have completed a full 360° circle.
5. Repeat with other fans (if available) until you reach the target count.
6. Save all images to: `Fan_Sticker_Dataset_V2/STICKER_PRESENT/`

## 5. How to Capture STICKER_MISSING Images
1. Take a fan **without** a sticker (or remove it).
2. **Crucial:** You must mimic the exact same orientations used for the PRESENT class.
3. Start at 0° (wire pointing down/right). Take 2-3 photos.
4. Rotate the fan ~45°. Take 2-3 photos.
5. Continue rotating through the 360° circle.
6. Save all images to: `Fan_Sticker_Dataset_V2/STICKER_MISSING/`

## 6. Recommended Number of Images
* **STICKER_PRESENT:** ~120+ real images (~15 images per 45° angle)
* **STICKER_MISSING:** ~120+ real images (~15 images per 45° angle)
* **Total:** ~240+ images.

## 7. Naming Convention
You can let your camera/phone auto-generate the filenames (e.g., `IMG_001.jpg`), but ensure they are saved directly into their respective class folders. Avoid generic names like `1.jpg` if mixing batches.

## 8. Examples of Good vs. Bad Dataset Collection

### ❌ BAD (The V1 Mistake)
- **PRESENT:** All photos have the fan upright.
- **MISSING:** All photos have the fan upside down.
*Result: The model learns orientation, not sticker presence.*

### ❌ BAD (Synthetic Cheating)
- Taking 1 upright photo and using software to rotate it 8 times.
*Result: The model overfits to the lighting/shadows of that single photo.*

### ✅ GOOD (The V2 Goal)
- **PRESENT:** 15 photos upright, 15 at 90°, 15 at 180°, etc.
- **MISSING:** 15 photos upright, 15 at 90°, 15 at 180°, etc.
*Result: Orientation is identical across classes. The model is forced to look at the center sticker to find the difference.*

## 9. How to Avoid Orientation Bias
**The Golden Rule:** For every orientation angle where you capture a STICKER_PRESENT image, you must capture a STICKER_MISSING image at that exact same angle. Do not allow one class to "own" a specific rotation.
