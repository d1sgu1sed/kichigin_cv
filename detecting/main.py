import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from collections import defaultdict
from pathlib import Path

def add_bottom_line(region_image):
    result_image = region_image.copy()
    result_image[-1, :] = 1
    return result_image

def is_A(region):
    image_with_line = add_bottom_line(region.image)
    labeled_image = label(image_with_line)
    new_region = regionprops(labeled_image)[0]
    return new_region.euler_number == -1

def is_W(region):
    h, w = region.image.shape
    center_area = region.image[h // 4 : -h // 4, w // 4 : -w // 4]
    gaps_count = np.sum(center_area == 0)
    return gaps_count > 5

def extractor(region):
    area_ratio = np.sum(region.image) / region.image.size
    perimeter_ratio = region.perimeter / region.image.size
    cy, cx = region.local_centroid
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    euler_number = region.euler_number
    eccentricity = region.eccentricity
    has_vertical_line = np.sum(np.mean(region.image, axis=0) == 1) > 3
    return np.array([area_ratio, perimeter_ratio, cy, cx, euler_number, eccentricity, has_vertical_line])

def classificator(region, classes):
    detected_class = None
    features = extractor(region)
    min_distance = float("inf")

    for cls, class_features in classes.items():
        dist = distance(features, class_features)
        if dist < min_distance:
            detected_class = cls
            min_distance = dist

    if detected_class in {"8", "B"}:
        if np.any(region.image[:, :2] == 0):
            detected_class = "8"
        else:
            detected_class = "B"

    if detected_class in {"A", "0"}:
        detected_class = "A" if is_A(region) else "0"

    if detected_class in {"*", "W"}:
        detected_class = "W" if is_W(region) else "*"

    return detected_class

def distance(v1, v2):
    return np.linalg.norm(v1 - v2)

image = plt.imread("alphabet.png")[:, :, :3].mean(axis=2)
image[image > 0] = 1

labeled_objects = label(image)
object_count = np.max(labeled_objects)

template = plt.imread("alphabet-small.png")[:, :, :3].mean(axis=2)
template[template < 1] = 0
template = np.logical_not(template)

labeled_template_objects = label(template)
template_count = np.max(labeled_template_objects)

regions = regionprops(labeled_template_objects)

classes = {
    "8": extractor(regions[0]),
    "0": extractor(regions[1]),
    "A": extractor(regions[2]),
    "B": extractor(regions[3]),
    "1": extractor(regions[4]),
    "W": extractor(regions[5]),
    "X": extractor(regions[6]),
    "*": extractor(regions[7]),
    "/": extractor(regions[8]),
    "-": extractor(regions[9]),
}

symbols_count = defaultdict(int)

for i, region in enumerate(regionprops(labeled_objects)):
    symbol = classificator(region, classes)
    symbols_count[symbol] += 1

print(symbols_count)
