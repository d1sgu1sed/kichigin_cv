import numpy as np
import cv2
import os

def init_images(dir):
    images = []
    for i in range(1, 13):
        image_dir = os.path.join(dir, "images", f"img ({i}).jpg")
        images.append(cv2.imread(image_dir))
        if images[i - 1] is None:
            print(f"Error by {image_dir}")
            exit(-1)
            
    return images

img_pencils = {}
pencils_counter = 0

main_dir = os.path.dirname(os.path.abspath(__file__))
images = init_images(main_dir)

for i, image in enumerate(images):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 192, cv2.THRESH_OTSU)
    img_binary = cv2.bitwise_not(binary)

    unique = np.unique(img_binary)
    img_binary[img_binary != unique[1]] = 0
    img_binary = img_binary[:, :img_binary.shape[1] - 50]

    edged = cv2.Canny(img_binary, 0, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    pencils_count = 0
    
    for cont in contours:
        p = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.02 * p, True)
        
        for j in range(1, len(approx)):
            dx = (approx[j][0][0] - approx[j-1][0][0]) ** 2
            dy = (approx[j][0][1] - approx[j-1][0][1]) ** 2
            edge_length = (dx + dy) ** 0.5  
            
            if edge_length >= 2500:
                pencils_count += 1
                pencils_counter += 1

    img_pencils[f"img ({i})"] = pencils_count

for image_name, count in img_pencils.items():
    print(f"{image_name}: {count} pencil(s)")

print(f"Pencils count: {pencils_counter}")