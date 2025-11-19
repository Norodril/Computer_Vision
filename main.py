input_path = "data/input.jpg"
output_dir = "output"
resize_width = 800
canny_low = 100
canny_high = 200
threshold_val = 120
'''
from config import *
from processing.loader import load_image
from processing.preprocess import resize, to_gray, blur
from processing.edges import detect_edges
from processing.contours import find_contours, draw_contours
from processing.segment import threshold, erode, dilate
from utils.io import ensure_dir, save

img = load_image(input_path)
img_resized = resize(img, resize_width)
gray = to_gray(img_resized)
blurred = blur(gray)
edges = detect_edges(blurred, canny_low, canny_high)
contours = find_contours(edges)
contoured_img = draw_contours(img_resized, contours)
threshed = threshold(gray, threshold_val)
eroded = erode(threshed)
dilated = dilate(eroded)

ensure_dir(output_dir)
save(output_dir + "/1_resized.jpg", img_resized)
save(output_dir + "/2_gray.jpg", gray)
save(output_dir + "/3_blur.jpg", blurred)
save(output_dir + "/4_edges.jpg", edges)
save(output_dir + "/5_contours.jpg", contoured_img)
save(output_dir + "/6_thresh.jpg", threshed)
save(output_dir + "/7_erode.jpg", eroded)
save(output_dir + "/8_dilate.jpg", dilated)
'''

import time
import sys
import json
from config import *
from processing.loader import load_image
from processing.preprocess import resize, to_gray, blur
from processing.edges import detect_edges
from processing.contours import find_contours, draw_contours
from processing.segment import threshold, erode, dilate
from utils.io import ensure_dir, save

def pipeline(path):
    t0 = time.time()
    img = load_image(path)
    t1 = time.time()
    img_resized = resize(img, resize_width)
    gray = to_gray(img_resized)
    blurred = blur(gray)
    edges = detect_edges(blurred, canny_low, canny_high)
    contours = find_contours(edges)
    contoured_img = draw_contours(img_resized, contours)
    threshed = threshold(gray, threshold_val)
    eroded = erode(threshed)
    dilated = dilate(eroded)
    merged = cv2.merge([gray, edges, threshed])
    t2 = time.time()

    stats = {
        "load_time": round(t1 - t0, 4),
        "process_time": round(t2 - t1, 4),
        "total_time": round(t2 - t0, 4),
        "contours_found": len(contours),
        "width": img_resized.shape[1],
        "height": img_resized.shape[0]
    }

    ensure_dir(output_dir)
    save(output_dir + "/1_resized.jpg", img_resized)
    save(output_dir + "/2_gray.jpg", gray)
    save(output_dir + "/3_blur.jpg", blurred)
    save(output_dir + "/4_edges.jpg", edges)
    save(output_dir + "/5_contours.jpg", contoured_img)
    save(output_dir + "/6_thresh.jpg", threshed)
    save(output_dir + "/7_erode.jpg", eroded)
    save(output_dir + "/8_dilate.jpg", dilated)
    save(output_dir + "/9_merged.jpg", merged)

    with open(output_dir + "/stats.json", "w") as f:
        json.dump(stats, f)

    return stats

def batch_mode(paths):
    results = {}
    for p in paths:
        r = pipeline(p)
        results[p] = r
    with open(output_dir + "/batch_summary.json", "w") as f:
        json.dump(results, f)
    return results

def parse_args():
    if len(sys.argv) == 1:
        return [input_path]
    return sys.argv[1:]

paths = parse_args()

if len(paths) == 1:
    s = pipeline(paths[0])
    print(json.dumps(s, indent=2))
else:
    b = batch_mode(paths)
    print(json.dumps(b, indent=2))