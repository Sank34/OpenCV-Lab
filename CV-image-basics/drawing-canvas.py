import cv2 as cv
import numpy as np

img = np.ones((600,800,3), dtype=np.uint8) * 255 # canvas

cv.putText(
    img,
    "Person: 99%",
    (50,30),
    cv.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0,0,0),
    3
)
cv.putText(
    img,
    "Distance: 1.4m",
    (200,30),
    cv.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0,0,0),
    3
)
cv.rectangle(
    img,
    (50,50),
    (300,250),
    (0,255,0),
    3
)

cv.circle(
    img,
    (300,400),
    80,
    (255,255,0),
    3
)
cv.imshow('Canvas', img)
cv.waitKey(0)
cv.destroyAllWindows()