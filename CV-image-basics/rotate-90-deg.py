import cv2 as cv

img = cv.imread('ratunel.png')

rot90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

rot90_ccw = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

rot180 = cv.rotate(img, cv.ROTATE_180)

cv.imshow('90 deg', rot90)
cv.imshow('90 deg ccw', rot90_ccw)
cv.imshow('180 deg', rot180)
cv.imshow('actual ratunel', img)
cv.waitKey(0)
cv.destroyAllWindows()