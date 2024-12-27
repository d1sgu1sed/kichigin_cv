import cv2
import time

cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

camera = cv2.VideoCapture(0)

lower_blue = (100, 100, 100)
upper_blue = (130, 255, 255)

D = 0.09
prev_time = time.time()
curr_time = time.time()
r = 1

trajectory, speed_values = [], []

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_write = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('balls.mp4', vid_write, 20.0, (frame_width, frame_height))

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    curr_time = time.time()
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_count = len(contours)

    if ball_count > 0:
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        trajectory.append((int(x), int(y)))
        if len(trajectory) > 10:
            trajectory.pop(0)
        if r > 20:
            cv2.circle(frame, (int(x), int(y)), 3, (120, 50, 0), -1)
            cv2.circle(frame, (int(x), int(y)), int(r), (120, 50, 0), 2)

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i], trajectory[i-1], (0, 100, 30), i)

        time_diff = curr_time - prev_time
        if len(trajectory) >= 2:
            p1 = trajectory[-1]
            p2 = trajectory[-2]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            pxl_per_m = D / (2 * r)
            dist *= pxl_per_m
            speed = dist / time_diff
            
            speed_values.append(speed)
            if len(speed_values) > 100:
                speed_values.pop(0)

            cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0))
            cv2.putText(frame, f"Objects Count: {ball_count}", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0))
            
            prev_time = curr_time

    out.write(frame)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
out.release()
cv2.destroyAllWindows()