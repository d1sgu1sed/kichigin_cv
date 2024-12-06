import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

def init_images(dir):
    images = []
    for i in range(0, 100):
        image_dir = os.path.join(dir, "out", f"h_{i}.npy")
        image = np.load(image_dir)
        images.append(image)
    return images

def find_centroid(labeled_image, pos):
    coords = np.where(labeled_image == pos)
    return (int)(coords[0].mean()), (int)(coords[1].mean())

def add_circle(labeled_image, pos, circle):
    xl, yl = find_centroid(labeled_image, pos)
    circle[0].append(xl)
    circle[1].append(yl)


main_dir = os.path.dirname(os.path.abspath(__file__))
images = init_images(main_dir)
circles = [[[], []], [[], []], [[], []]]

labeled_image = label(images[0])
for i, circle in enumerate(circles):
    add_circle(labeled_image, i + 1, circle)

for image in images[1:]:
    labeled_image = label(image)
    for lbl in range(1, 4):
        pos = np.where(labeled_image == lbl)
        last_pos = {i: (None, None) for i in range(3)}
        for x, y in zip(*pos):
            for i, circle in enumerate(circles):
                last_x, last_y = last_pos[i]
                if last_x == x and last_y == y:
                    add_circle(labeled_image, lbl, circle)
            for i in range(3):
                if circles[i][0]:
                    last_pos[i] = (circles[i][0][-1], circles[i][1][-1])

for i, circle in enumerate(circles):
    plt.plot(circle[0], circle[1], label=f'circle №{i + 1}')

plt.legend()
plt.show()