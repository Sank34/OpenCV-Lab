import cv2 as cv

img = cv.imread('ratunel.png')

blurred = cv.GaussianBlur(img, (15,15), 0)

# larger kernel => more blur

# median blur
# kernel size MUST be odd

median = cv.medianBlur(img, 15)

cv.imshow('actual ratunel', img)
cv.imshow('blurred ratunel', blurred)
cv.imshow('median blurred ratunel', median)
cv.waitKey(0)
cv.destroyAllWindows()