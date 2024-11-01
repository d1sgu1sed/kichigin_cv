import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.ndimage import binary_erosion

image = np.load("ps.npy")

struct1 = [[0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1],
          [1,1, 1, 1, 1, 1]]

struct2 = [[0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 1, 1],
          [1, 1, 0, 0, 1, 1],
          [1, 1, 1, 1, 1, 1],
          [1,1, 1, 1, 1, 1]]

struct3 = [[1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 1, 1],
          [1, 1, 0, 0, 1, 1],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0]]

struct4 = [[1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 0, 0]]

struct5 = [[0, 0, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1],
          [0, 0, 0, 0, 1, 1],
          [0, 0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1]]

structs = [struct1, struct2, struct3, struct4, struct5]

for i, struct in enumerate(structs):
    print(f"Структур №{i + 1}: {np.max(label(binary_erosion(image, struct))) - 
                               (0 if struct not in [struct2, struct3] else 
                               np.max(label(binary_erosion(image, struct1))))}"
    )
plt.imshow(image)
plt.title(f'Всего {np.max(label(image))} элементов')
plt.show()