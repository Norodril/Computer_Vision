import cv2

def find_contours(img):
    c, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return c

def draw_contours(orig, contours):
    return cv2.drawContours(orig.copy(), contours, -1, (0, 255, 0), 2)