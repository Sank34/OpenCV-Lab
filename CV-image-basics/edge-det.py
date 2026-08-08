import cv2 as cv

image = cv.imread('ratunel.png')

# conv to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#edges
edges = cv.Canny(gray, 100, 200)

#save!
cv.imwrite('edges.png', edges)