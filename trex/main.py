import mss
import numpy as np
import cv2
import pyautogui
import time

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return thresh

def detect_obstacles_and_pterodactyls(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    pterodactyls = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < w < 120 and 10 < h < 120:
            if y < 50:
                pterodactyls.append((x, y, w, h))
                break
            else:
                obstacles.append((x, y, w, h))
                break

    return obstacles, pterodactyls

def adjust_monitor_speed(speed):
    base_width = 80
    return {
        "top": 324,
        "left": 750,
        "width": base_width + int(speed * 0.16), 
        "height": 120,
    }
def handle_actions(obstacles, pterodactyls, fl):
    if obstacles:
        pyautogui.press("space") 
    elif pterodactyls:
        pyautogui.keyDown("down")
        fl = True
    else:
        if fl:
            pyautogui.keyDown("down")
            fl = False
        pyautogui.keyUp("down") 

speed = 0  
fl = False
with mss.mss() as sct:
    monitor = adjust_monitor_speed(speed)

    print("Запуск через 5 секунд...")
    time.sleep(5)

    while True:
        screenshot = np.array(sct.grab(monitor))

        processed_image = process_image(screenshot)

        obstacles, pterodactyls = detect_obstacles_and_pterodactyls(processed_image)

        handle_actions(obstacles, pterodactyls, fl)

        # Увеличение скорости каждые 7 циклов
        if speed % 7 == 0:
            monitor = adjust_monitor_speed(speed)

        speed += 1

