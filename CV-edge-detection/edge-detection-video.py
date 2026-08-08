import cv2 as cv
import numpy as np

def nothing(x):
    pass

# trackbars
cv.namedWindow('edge')
cv.createTrackbar('threshold1', 'edge', 2000,5000,nothing)
cv.createTrackbar('threshold2', 'edge', 4000,5000,nothing)

cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break;

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    thrs1 = cv.getTrackbarPos('threshold1', 'edge')
    thrs2 = cv.getTrackbarPos('threshold2', 'edge')

    edges = cv.Canny(gray, thrs1, thrs2, apertureSize=5)

    vis = frame.copy()
    vis = np.uint8(vis / 2.0)
    vis[edges != 0] = (0,255,0)

    cv.imshow('edge', vis)

    if cv.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()