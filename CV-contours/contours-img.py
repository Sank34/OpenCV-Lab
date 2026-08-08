import cv2 as cv

img = cv.imread('ratunel.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(
    gray,
    100,
    255,
    cv.THRESH_BINARY
)

contours, hierachy = cv.findContours(
    thresh,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

output = img.copy()

cv.drawContours(
    output,
    contours,
    -1,
    (0,255,0),
    2
)

cv.imshow('actual ratunel', img)
cv.imshow('contours', output)
cv.waitKey(0)
cv.destroyAllWindows()