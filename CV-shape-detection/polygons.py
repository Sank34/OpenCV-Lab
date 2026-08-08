import cv2 as cv
import math

img = cv.imread('shapes-new.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(
    gray,
    0, # 240
    255,
    cv.THRESH_BINARY_INV + cv.THRESH_OTSU
)

contours, hierarchy = cv.findContours(
    thresh,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

output = img.copy()

for contour in contours:
    area = cv.contourArea(contour)

    if area < 500:
        continue

    perimeter = cv.arcLength(contour, True) #forma inchisa
    if perimeter == 0:
        continue

    approx = cv.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    vertices = len(approx)
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    shape = "none"
    if vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        # patrat vs dreptunghi
        x,y,w,h = cv.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 0.90 <= aspect_ratio <= 1.10:
            shape = "square"
        else:
            shape = "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    elif vertices == 6:
        shape = "hexagon"
    elif circularity > 0.80:
        shape = "circle"

    x,y,w,h = cv.boundingRect(approx)

    cv.drawContours(
        output,
        [approx],
        -1,
        (0,255,0),
        2
    )

    cv.putText(
        output,
        shape,
        (x, max(y-10,20)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

cv.imshow("thresh", thresh)
cv.imshow("shape detection", output)
cv.waitKey(0)
cv.destroyAllWindows()