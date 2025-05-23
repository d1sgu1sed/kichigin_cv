import cv2
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from skimage import draw


path = Path(__file__).parent
model_path = path / "facial_best.pt"

cap = cv2.VideoCapture(0)
model = YOLO(model_path)

while cap.isOpened():
    ret, frame = cap.read()

    result = model(frame)[0]
    masks = result.masks
    annotated = result.plot()
    oranges = cv2.imread(str(path / 'oranges.png'))
    hsv_oranges = cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV)

    lower = np.array((10, 239, 200))
    upper = np.array((50, 255, 255))

    orange_mask = cv2.inRange(hsv_oranges, lower, upper)
    orange_mask = cv2.dilate(orange_mask, np.ones((7, 7)))
    # cv2.imshow('orange_mask', orange_mask)
    # cv2.imshow('hsv_oranges', hsv_oranges)

    contours, _ = cv2.findContours(orange_mask,  cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=cv2.contourArea)
    m = cv2.moments(sorted_contours[-1])
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    print(cx, cy)

    bbox = cv2.boundingRect(sorted_contours[-1])

    if not(masks):
        continue
    global_mask = masks[0].data.cpu().numpy()[0, :, :]

    for mask in masks:
        global_mask += mask.data.cpu().numpy()[0, :, :]

    rr, cc = draw.disk((4, 4), 4)
    struct = np.zeros((9, 9), np.uint8)
    struct[rr, cc] = 1

    global_mask = cv2.resize(global_mask, (frame.shape[1], frame.shape[0])).astype("uint8")
    global_mask = cv2.GaussianBlur(global_mask, (7, 7), 0)
    global_mask = cv2.dilate(global_mask, struct)
    global_mask = global_mask.reshape(frame.shape[0], frame.shape[1], 1)

    parts = (frame * global_mask).astype("uint8")

    pos = np.where(global_mask > 0)
    min_y, max_y = int(np.min(pos[0]) * 0.7), int(np.max(pos[0]) * 1.1)
    min_x, max_x = int(np.min(pos[1]) * 0.7), int(np.max(pos[1]) * 1.1)
    global_mask = global_mask[min_y:max_y, min_x:max_x]
    parts = parts[min_y:max_y, min_x:max_x]

    resized_parts = cv2.resize(parts, (bbox[2], bbox[3]))
    resized_mask = cv2.resize(global_mask, (bbox[2], bbox[3])) * 255

    # oranges[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = resized_parts
    x, y, w, h = bbox
    roi = oranges[y:y+h, x:x+w]
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(resized_mask))
    combined = cv2.add(bg, resized_parts)
    oranges[y:y+h, x:x+w] = combined

    cv2.imshow("Image", oranges)
    # cv2.imshow("Mask", parts)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
