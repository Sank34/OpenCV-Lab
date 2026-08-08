import cv2 as cv

image = cv.imread('ratunel.png')

# display the image

cv.imshow('New Image of Ratunel', image)
cv.waitKey(0)
cv.destroyAllWindows()