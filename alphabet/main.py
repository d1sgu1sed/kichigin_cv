import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops, euler_number

def recognize(region):
    if region.image.mean() == 1.0:
        return "-"

    enumber = euler_number(region.image, connectivity=2)

    if enumber == -1:
        half_col_mean = np.mean(region.image[:, :region.image.shape[1] // 2], axis=0)
        if np.sum(half_col_mean == 1) > 3:
            return "B"
        return "8"

    elif enumber == 0:
        image = region.image.copy()

        if 1 in image.mean(axis=0)[:2]:
            image[len(image) // 2, :] = 1
            image_regions = regionprops(label(image))
            if image_regions[0].euler_number == -1:
                return "D"
            return "P"

        image[-1, :] = 1
        if euler_number(image) == -1:
            return "A"
        return "0"

    if 1 in region.image.mean(axis=0):
        return "1"

    image = region.image.copy()
    image[[0, -1], :] = 1
    image_regions = regionprops(label(image))
    euler = image_regions[0].euler_number

    if euler == -1:
        return "X"
    elif euler == -2:
        return "W"

    if region.eccentricity > 0.5:
        return "/"
    return "*"


image = plt.imread("./symbols.png")[:, :, :3].mean(axis=2)
image[image > 0] = 1

labeled_image = label(image)
regions = regionprops(labeled_image)

result = {}

for region in regions:
    symbol = recognize(region)
    result[symbol] = result.get(symbol, 0) + 1

print("Частотный словарь:", result)
print(f"Всего символов на картинке: {labeled_image.max()}")

