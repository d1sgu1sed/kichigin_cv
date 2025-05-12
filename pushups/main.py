from pathlib import Path
import time
from typing import Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

ROOT_DIR = Path(__file__).parent
MODEL_FILE = ROOT_DIR / "yolo11n-pose.pt"
STREAM_URL = "http://192.168.0.102:8080/video"
OUTPUT_VIDEO = ROOT_DIR / "out.avi"
FRAME_SIZE = (800, 600)
FPS_TARGET = 10
ELBOW_THRESHOLD = 140
RESET_TIMEOUT = 20

def _distance(p1, p2):
    dx, dy = (int(p1[0]) - int(p2[0]), int(p1[1]) - int(p2[1]))
    return float(np.hypot(dx, dy))


def _angle(a, b, c) -> float:
    d = np.arctan2(c[1] - b[1], c[0] - b[0])
    e = np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = np.degrees(d - e)
    ang = (ang + 360) % 360
    return 360 - ang if ang > 180 else ang

def _compute_elbow_angles(frame, kpts):
    try:
        l_sh, r_sh = kpts[5], kpts[6]
        l_el, r_el = kpts[7], kpts[8]
        l_wr, r_wr = kpts[9], kpts[10]

        a_left = _angle(l_sh, l_el, l_wr)
        a_right = _angle(r_sh, r_el, r_wr)

        cv2.putText(
            frame, f"Left Elbow: {a_left:5.1f}",
            (int(l_el[0]) + 10, int(l_el[1]) + 10),
            cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2
        )
        cv2.putText(
            frame, f"Right Elbow: {a_right:5.1f}",
            (int(r_el[0]) + 10, int(r_el[1]) + 30),
            cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2
        )
        return a_left, a_right
    except (IndexError, ZeroDivisionError):
        return None, None



model = YOLO(MODEL_FILE)
cap = cv2.VideoCapture(STREAM_URL)
writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"MJPG"),
    FPS_TARGET,
    FRAME_SIZE
)

last_frame_time = time.time()
last_push_time = time.time()
counter, arms_up = 0, True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = max(now - last_frame_time, 1e-6)
    last_frame_time = now
    cv2.putText(frame, f"FPS: {1/dt:4.1f}",
                (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 25), 1)

    result = model(frame)[0]
    if not result.keypoints:
        cv2.imshow("Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()

    angles = _compute_elbow_angles(annotated, result.keypoints.xy[0].tolist())
    if all(a is not None for a in angles):
        left, right = angles
        if arms_up and left < ELBOW_THRESHOLD and right < ELBOW_THRESHOLD:
            counter += 1
            last_push_time = now
            arms_up = False
        elif left > ELBOW_THRESHOLD and right > ELBOW_THRESHOLD:
            arms_up = True

    if now - last_push_time > RESET_TIMEOUT:
        counter = 0

    cv2.putText(annotated, f"Count: {counter}",
                (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (25, 255, 25), 2)

    cv2.imshow("Pose", annotated)
    writer.write(cv2.resize(annotated, FRAME_SIZE))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
