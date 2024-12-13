from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from collections import Counter

def merge_close_hues(hues, threshold=0.05):
    merged = Counter()
    sorted_hues = sorted(hues.items())
    temp_hue, temp_count = sorted_hues[0]

    for hue, count in sorted_hues[1:]:
        if abs(hue - temp_hue) < threshold:
            temp_count += count
        else:
            merged[temp_hue] = temp_count
            temp_hue, temp_count = hue, count

    merged[temp_hue] = temp_count
    return merged

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
    fl = True
    if recognized == 'circles':
        crcls_colors.append(color)
    else:
        rects_colors.append(color)


total_objects = objects["circles"] + objects["rects"]

crcls_hue_counts = Counter([round(color, 2) for color in crcls_colors])
crcls_hue_counts = merge_close_hues(crcls_hue_counts)
rects_hue_counts = Counter([round(color, 2) for color in rects_colors])
rects_hue_counts = merge_close_hues(rects_hue_counts)


print(f"Total objects: {total_objects}")
print(f"Circles: {objects['circles']}")
print(f"Rectangles: {objects['rects']}")
print("\nCircle counts by hue:")
for hue, count in crcls_hue_counts.items():
    print(f"Hue {hue}: {count}")

print("\nRectangle counts by hue:")
for hue, count in rects_hue_counts.items():
    print(f"Hue {hue}: {count}")




