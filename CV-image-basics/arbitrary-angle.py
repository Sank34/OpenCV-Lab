import cv2 as cv

img = cv.imread('ratunel.png')
h,w = img.shape[:2]

# rot matrix pt 45 deg
center = (w//2,h//2)
angle = 45
scale = 1.0
rot_mat = cv.getRotationMatrix2D(center,angle,scale)

rotated = cv.warpAffine(img,rot_mat,(w,h))

cv.imshow('rotated', rotated)
cv.imshow('actual ratunel', img)
cv.waitKey(0)
cv.destroyAllWindows()