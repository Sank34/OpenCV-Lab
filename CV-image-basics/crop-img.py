import cv2 as cv

img = cv.imread('ratunel.png')

# slicing
cropped = img[100:180, 100:200]

# ROI
x, y, w, h = 100,50, 100,100
roi = img[y:y+h, x:x+w]

cv.imshow('cropped 1', cropped)
cv.imshow('actual ratunel', img)
cv.imshow('roi', roi)
cv.waitKey(0)
cv.destroyAllWindows()