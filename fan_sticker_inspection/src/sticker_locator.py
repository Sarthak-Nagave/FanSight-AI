"""
Lightweight OpenCV-based sticker locator for visualization only.
This does NOT affect the YOLO classification result.
It finds the likely sticker region to draw a tight visual marker.
"""
import cv2
import numpy as np


def locate_sticker(image_bgr):
    """
    Attempt to locate the circular sticker on the fan hub.
    
    The sticker is a small yellow/white circular label at the center of the
    black fan hub. We detect it by looking for bright yellowish regions
    that are roughly circular and appropriately sized.
    
    Returns:
        dict or None: {'cx': int, 'cy': int, 'radius': int} if found, else None.
    """
    h, w = image_bgr.shape[:2]
    
    # Convert to HSV for color-based filtering
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # The sticker appears as a bright yellow/white circle.
    mask_yellow = cv2.inRange(hsv, (15, 50, 150), (40, 255, 255))
    mask_light = cv2.inRange(hsv, (0, 20, 180), (45, 255, 255))
    mask = cv2.bitwise_or(mask_yellow, mask_light)
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by size and circularity
    min_area = (min(h, w) * 0.02) ** 2
    max_area = (min(h, w) * 0.25) ** 2
    
    best_candidate = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < 0.3:
            continue
        
        score = circularity * np.sqrt(area)
        
        if score > best_score:
            best_score = score
            best_candidate = contour
    
    if best_candidate is None:
        return None
    
    (cx, cy), radius = cv2.minEnclosingCircle(best_candidate)
    cx, cy, radius = int(cx), int(cy), int(radius)
    radius = int(radius * 1.15)
    
    return {'cx': cx, 'cy': cy, 'radius': radius}


def get_expected_sticker_region(image_bgr):
    """
    When the sticker is MISSING, estimate the expected sticker area
    by finding the fan hub center using HoughCircles on the dark fan body.
    
    Falls back to finding the largest dark contour's centroid,
    then to the image center.
    
    Returns:
        dict: {'cx': int, 'cy': int, 'radius': int}
    """
    h, w = image_bgr.shape[:2]
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Try HoughCircles to find circular fan opening
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Scale parameters based on image size
    min_radius = int(min(h, w) * 0.05)
    max_radius = int(min(h, w) * 0.25)
    min_dist = int(min(h, w) * 0.15)
    
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min_dist,
        param1=100, param2=50,
        minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Pick the circle closest to the image center
        img_cx, img_cy = w // 2, h // 2
        best_circle = None
        best_dist = float('inf')
        for c in circles[0, :]:
            d = np.sqrt((int(c[0]) - img_cx) ** 2 + (int(c[1]) - img_cy) ** 2)
            if d < best_dist:
                best_dist = d
                best_circle = c
        
        if best_circle is not None:
            cx, cy, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            sticker_radius = max(int(r * 0.30), 20)
            return {'cx': cx, 'cy': cy, 'radius': sticker_radius}
    
    # Fallback: find largest dark blob excluding very large ones (background objects)
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter out contours that are too large (e.g. monitor, table edge)
        max_contour_area = h * w * 0.25
        min_contour_area = h * w * 0.005
        
        valid_contours = [c for c in contours 
                          if min_contour_area < cv2.contourArea(c) < max_contour_area]
        
        if valid_contours:
            # Pick the most circular one
            best = None
            best_circ = 0
            for c in valid_contours:
                area = cv2.contourArea(c)
                perim = cv2.arcLength(c, True)
                if perim == 0:
                    continue
                circ = 4 * np.pi * area / (perim * perim)
                if circ > best_circ:
                    best_circ = circ
                    best = c
            
            if best is not None:
                M = cv2.moments(best)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    fan_radius = int(np.sqrt(cv2.contourArea(best) / np.pi))
                    sticker_radius = max(int(fan_radius * 0.25), 20)
                    return {'cx': cx, 'cy': cy, 'radius': sticker_radius}
    
    # Final fallback: image center
    return {
        'cx': w // 2,
        'cy': h // 2,
        'radius': int(min(h, w) * 0.05)
    }
