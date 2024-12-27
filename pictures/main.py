import cv2
import numpy as np

VIDEO_PATH = 'output.avi'
IMAGE_PATH = 'kichigin_iv.png'
THRESHOLD = 0.2
MATCH_COUNT_RANGE = (350, 1000)

template = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError(f"Изображение не найдено: {IMAGE_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Видео не найдено: {VIDEO_PATH}")

frame_count = 0
image_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= THRESHOLD)

    match_count = len(loc[0])
    if MATCH_COUNT_RANGE[0] <= match_count <= MATCH_COUNT_RANGE[1]:
        image_count += 1

    frame_count += 1

cap.release()

print(f"Количество кадров с изображением: {image_count}")
