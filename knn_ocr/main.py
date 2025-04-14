import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
import matplotlib.pyplot as plt

def fix_label(label: str) -> str:
    if label.startswith('s') and len(label) > 1:
        label = label[1:]
    return label

def load_training_data(train_dir, image_size=(20, 20)):
    X, y = [], []
    for root, dirs, files in os.walk(train_dir):
        label = os.path.basename(root)
        if label == os.path.basename(train_dir):
            continue
        
        label = fix_label(label)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                img_resized = cv2.resize(img_bin, image_size, interpolation=cv2.INTER_AREA)

                features = hog(img_resized, 
                               orientations=9, 
                               pixels_per_cell=(4, 4), 
                               cells_per_block=(2, 2), 
                               block_norm='L2-Hys')
                X.append(features)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

def train_knn(X, y, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X, y)
    return knn

def binarize_image(img_color):
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    
    lower_purple = np.array([110, 40, 40])
    upper_purple = np.array([170, 255, 255])
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    thr_bin = cv2.adaptiveThreshold(gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 31, 15)

    combined = cv2.bitwise_or(thr_bin, mask_purple)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    return opened

def group_bboxes_by_lines(bboxes, line_gap=30):
    if not bboxes:
        return []
    bboxes_sorted = sorted(bboxes, key=lambda b: (b[1], b[0]))
    lines = []
    current_line = [bboxes_sorted[0]]
    for i in range(1, len(bboxes_sorted)):
        (x, y, w, h) = bboxes_sorted[i]
        (px, py, pw, ph) = bboxes_sorted[i - 1]
    
        if abs(y - py) <= line_gap:
            current_line.append(bboxes_sorted[i])
        else:
            lines.append(current_line)
            current_line = [bboxes_sorted[i]]
    if current_line:
        lines.append(current_line)

    for idx, line_bboxes in enumerate(lines):
        lines[idx] = sorted(line_bboxes, key=lambda b: b[0])
    return lines

def extract_and_predict_text(image_path, knn, image_size=(20, 20), debug=False):
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"Ошибка чтения изображения: {image_path}")
        return ""

    img_bin = binarize_image(img_color)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    bboxes = [b for b in bboxes if b[2] > 2 and b[3] > 2]

    if debug:
        dbg_img = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in bboxes:
            cv2.rectangle(dbg_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        plt.figure(figsize=(10,4))
        plt.title(f"{os.path.basename(image_path)} bounding boxes")
        plt.imshow(dbg_img[..., ::-1])
        plt.axis('off')
        plt.show()

    lines = group_bboxes_by_lines(bboxes, line_gap=30)
    
    recognized_lines = []
    for line_bboxes in lines:
        line_text = ""
    
        if len(line_bboxes) > 1:
            distances = [
                line_bboxes[i][0] - (line_bboxes[i-1][0] + line_bboxes[i-1][2])
                for i in range(1, len(line_bboxes))
            ]
            mean_dist = np.mean(distances)
        else:
            mean_dist = 0
        
        for i, (x, y, w, h) in enumerate(line_bboxes):
            if i > 0:
                gap = x - (line_bboxes[i-1][0] + line_bboxes[i-1][2])
                if mean_dist > 0 and gap > 1.8 * mean_dist:
                    line_text += " "

            roi = img_bin[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, image_size, interpolation=cv2.INTER_AREA)

            features = hog(roi_resized, 
                           orientations=9, 
                           pixels_per_cell=(4, 4), 
                           cells_per_block=(2, 2), 
                           block_norm='L2-Hys')
            pred_symbol = knn.predict(features.reshape(1, -1))[0]
            line_text += pred_symbol
        recognized_lines.append(line_text)

    recognized_text = "\n".join(recognized_lines)
    return recognized_text

def process_all_images(train_dir="train", root_dir=".", debug=False):
    X, y = load_training_data(train_dir)
    knn = train_knn(X, y)

    results = {}
    for file in sorted(os.listdir(root_dir)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and not file.startswith("train"):
            path = os.path.join(root_dir, file)
            text = extract_and_predict_text(path, knn, debug=debug)
            print(f"Изображение: {file} => Распознанный текст:\n{text}")
            print("-" * 50)
            results[file] = text
    return results

results = process_all_images(train_dir="./task/train", root_dir="./task", debug=True)

