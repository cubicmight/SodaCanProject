import os
import cv2
import numpy as np

basedir = os.path.abspath(os.path.dirname(__file__))

image_file_name = "videos/IMG_7834.MOV"
full_image_path = os.path.join(basedir, image_file_name)
if not os.path.exists(full_image_path):
    print("cannot find image")
    exit(-1)

cap = cv2.VideoCapture(full_image_path)


def draw_circle():
    global circles
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


def find_circle():
    global output, gray, circles
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0);
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 11, 3.5)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)
    gray = cv2.dilate(gray, kernel, iterations=1)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=45, minRadius=145, maxRadius=290)


while True:
    ret, frame = cap.read()
    if not ret:
        exit(-1)

    find_circle()

    if circles is not None:
        draw_circle()

        cv2.imshow('gray', gray)
    cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
