import cv2

def threshold(img, val):
    _, t = cv2.threshold(img, val, 255, cv2.THRESH_BINARY)
    return t

def erode(img):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(img, k)

def dilate(img):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(img, k)