import cv2 as cv

img = cv.imread('ratunel.png')

flipped_h = cv.flip(img,1)
flipped_v = cv.flip(img,0)
flipped_both = cv.flip(img, -1)

cv.imshow('actual ratunel', img)
cv.imshow('flipped ratunel horiz', flipped_h)
cv.imshow('flipped ratunel vert', flipped_v)
cv.imshow('flipped ratunel both', flipped_both)
cv.waitKey(0)
cv.destroyAllWindows()