import cv2
import numpy as np

def fupdate(value):
    global flimit
    flimit = value

def supdate(value):
    global slimit
    slimit = value

rtsp_url = "rtsp://192.168.0.103:8080/h264_ulaw.sdp"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()

window_name = "client"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.createTrackbar("F", window_name, 100, 255, fupdate)
cv2.createTrackbar("S", window_name, 200, 255, supdate)

lower = (0, 70, 120)
upper = (255, 255, 255)

green_lower = (35, 50, 50)
green_upper = (85, 255, 255)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    combined_mask = cv2.bitwise_or(mask, green_mask)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    gray = cv2.GaussianBlur(mask, (7, 7), 0)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    count_cube = 0
    count_all = 0
    for contour in contours:
        if cv2.contourArea(contour) > 3000:
            rect = cv2.minAreaRect(contour)
            (x, y), (w, h), angle = rect

            aspect_ratio = min(w, h) / max(w, h)
            rect_area = w * h
            contour_area = cv2.contourArea(contour)

            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and aspect_ratio > 0.8 and contour_area / rect_area > 0.8:
                count_cube += 1
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            else:
                cv2.circle(frame, (int(x), int(y)), int(max(w, h) // 2), (255, 0, 0), 2)

            count_all += 1

    cv2.putText(frame, f"Count objects: {count_all}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Count spheres: {count_all - count_cube}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Count cubes: {count_cube}", (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    key = cv2.waitKey(100)
    if key == ord('q'):
        break

    cv2.imshow(window_name, frame)

cap.release()
cv2.destroyAllWindows()
