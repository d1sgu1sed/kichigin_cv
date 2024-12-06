from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from collections import Counter

im = plt.imread("balls_and_rects.png")
im_hsv = rgb2hsv(im)

binary = im.mean(2)
binary[binary > 0] = 1

labeled = label(binary)
regions = regionprops(labeled)

crcls_colors = []
rects_colors = []
objects = {
    "circles": 0,
    "rects": 0
}

for region in regions:
    recognized = 'circles' if region.eccentricity < 0.6 else 'rects'
    objects[recognized] += 1
    cy, cx = region.centroid
    color = im_hsv[int(cy), int(cx)][0]
    if recognized == 'circles':
        crcls_colors.append(color)
    else:
        rects_colors.append(color)

circle_hue_counts = Counter([round(color, 2) for color in crcls_colors])
rect_hue_counts = Counter([round(color, 2) for color in rects_colors])


total_objects = objects["circles"] + objects["rects"]
print(f"Total objects: {total_objects}")
print(f"Circles: {objects['circles']}")
print(f"Rectangles: {objects['rects']}")
print("\nCircle counts by hue:")
for hue, count in circle_hue_counts.items():
    print(f"Hue {hue}: {count}")

print("\nRectangle counts by hue:")
for hue, count in rect_hue_counts.items():
    print(f"Hue {hue}: {count}")




