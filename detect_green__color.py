import cv2
import numpy as np
import os
import sys

def detect_green(image_path, min_ratio=0.005, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERR] Cannot read:", image_path)
        return False, 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green range (tune if needed)
    lower = np.array([35, 60, 40], dtype=np.uint8)
    upper = np.array([85,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    green_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0]*img.shape[1]
    ratio = green_pixels / (total_pixels + 1e-6)
    is_present = ratio >= min_ratio

    print(f"[INFO] Green ratio: {ratio:.4%} => {'DETECTED' if is_present else 'NOT DETECTED'} (thr {min_ratio:.2%})")

    if debug:
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = img.copy()
        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite("green_mask.png", mask)
        cv2.imwrite("green_vis.png", vis)
        print("[INFO] Saved green_mask.png, green_vis.png")

    return is_present, ratio

if __name__ == "__main__":
    # Usage: python detect_green__color.py [image_path] [--debug] [--thr=0.003]
    img_path = "image.png"
    debug = False
    thr = 0.005
    for arg in sys.argv[1:]:
        if arg == "--debug":
            debug = True
        elif arg.startswith("--thr="):
            try:
                thr = float(arg.split("=",1)[1])
            except ValueError:
                pass
        else:
            img_path = arg

    if not os.path.exists(img_path):
        print("[ERR] File not found:", img_path)
        sys.exit(1)

    detect_green(img_path, min_ratio=thr, debug=debug)
    
API_KEY = "AIzaSyCjxNLBdRyJqyo5bJRT14sTC-nR9BzEOy0"