import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

def nothing(x):
    pass
cv.namedWindow('preview')
cv.createTrackbar('threshold1', 'preview', 2000, 5000, nothing)
cv.createTrackbar('threshold2', 'preview', 4000, 5000, nothing)
mode = "original"
while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv.GaussianBlur(frame, (35,35), 0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    t1 = cv.getTrackbarPos('threshold1', 'preview')
    t2 = cv.getTrackbarPos('threshold2', 'preview')
    edges = cv.Canny(gray, t1, t2, apertureSize=5)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('g'):
        mode = "blurred"
    elif key == ord('e'):
        mode = "gray"
    elif key == ord('b'):
        mode = "edges"
    elif key == ord('o'):
        mode = "original"

    if mode == "gray":
        output = gray
    elif mode == "blurred":
        output = blurred
    elif mode == "edges":
        output = edges
    else:
        output = frame

    cv.imshow('preview', output)

cap.release()
cv.destroyAllWindows()