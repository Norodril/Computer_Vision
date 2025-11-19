import cv2

def detect_edges(img, low, high):
    return cv2.Canny(img, low, high)
