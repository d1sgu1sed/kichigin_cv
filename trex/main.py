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
            if y < 50:  # Птеродактиль
                pterodactyls.append((x, y, w, h))
            else:  # Препятствие
                obstacles.append((x, y, w, h))

    return obstacles, pterodactyls

def adjust_monitor_speed(speed):
    base_width = 80
    return {
        "top": 324,
        "left": 749,
        "width": base_width + int(speed * 0.08),
        "height": 120,
    }

def handle_actions(obstacles, pterodactyls, fl):
    # print(fl)
    if fl:  # Отжать клавишу вниз, если нет угроз
        pyautogui.keyUp("down")
        fl = False
        return fl

    if obstacles:
        obstacle = min(obstacles, key=lambda obs: obs[0])
        if obstacle[0] < 150:  # Если препятствие близко
            pyautogui.press("space")  # Прыжок
            return fl

    if pterodactyls:
        pterodactyl = min(pterodactyls, key=lambda ptero: ptero[0])
        if pterodactyl[0] < 35:  # Если птеродактиль близко
            if not fl:
                pyautogui.keyDown("down")  # Присесть
                fl = True
        return fl
    return fl
    

def main():
    speed, timer = 0, 0
    fl = False
    with mss.mss() as sct:
        monitor = adjust_monitor_speed(speed)

        print("Запуск через 5 секунд...")
        time.sleep(5)

        while True:
            screenshot = np.array(sct.grab(monitor))
            processed_image = process_image(screenshot)

            obstacles, pterodactyls = detect_obstacles_and_pterodactyls(processed_image)

            fl = handle_actions(obstacles, pterodactyls, fl)

            # Увеличение скорости каждые 5 циклов
            if timer % 7 == 0:
                speed += 1
                monitor = adjust_monitor_speed(speed)

            # Отладочная информация
            debug_image = processed_image.copy()
            for x, y, w, h in obstacles:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for x, y, w, h in pterodactyls:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Debug", debug_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            timer += 1

if __name__ == "__main__":
    main()
