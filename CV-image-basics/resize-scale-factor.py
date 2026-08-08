import cv2 as cv

img = cv.imread('ratunel.png')

t = 0.5
print(img.shape)
width = int(img.shape[1] * t)
height = int(img.shape[0] * t)
res_scale = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)

fx, fy = 0.5, 0.5
resized_aspect_ratio = cv.resize(img, None, fx=fx, fy=fy, interpolation=cv.INTER_AREA)
cv.imshow('res scale img', res_scale)
cv.imshow('actual ratunel', img)
cv.imshow('res aspect ratio', resized_aspect_ratio)
cv.waitKey(0)
cv.destroyAllWindows()