from contextlib import closing

import cv2 as cv
import numpy as np

img = cv.imread('ratunel.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(
    gray,
    100,
    255,
    cv.THRESH_BINARY
)

# kernel = np.ones((5,5), np.uint8)
kernel = cv.getStructuringElement(
    cv.MORPH_RECT,
    (5,5)
)

erosion = cv.erode(
    thresh,
    kernel,
    iterations=5
)

dilation = cv.dilate(
    thresh,
    kernel,
    iterations=5
)
# open = erode + dilate
opening = cv.morphologyEx(
    thresh,
    cv.MORPH_OPEN,
    kernel
)

closing = cv.morphologyEx(
    thresh,
    cv.MORPH_CLOSE,
    kernel
)

cv.imshow('actual ratunel', img)
cv.imshow('gray ratunel', gray)
cv.imshow('thresh ratunel', thresh)
cv.imshow('erosion of ratunel', erosion)
cv.imshow('dilation of ratunel', dilation)
cv.imshow('opening', opening)
cv.imshow('closing', closing)
cv.waitKey(0)
cv.destroyAllWindows()