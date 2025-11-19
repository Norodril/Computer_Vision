import os
import cv2

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save(path, img):
    cv2.imwrite(path, img)