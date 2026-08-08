import cv2 as cv

img = cv.imread('ratunel.png')

print(img.shape)

resized = cv.resize(img, (120,80))

cv.imshow("resized ratunel", resized)
cv.imshow("actual ratunel", img)
cv.waitKey(0)
cv.destroyAllWindows()