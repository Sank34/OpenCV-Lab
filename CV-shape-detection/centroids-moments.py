import cv2 as cv
import math

img = cv.imread('shapes-new.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

otsu_score, thresh = cv.threshold(
    gray,
    0,
    255,
    cv.THRESH_BINARY_INV + cv.THRESH_OTSU
)
print(f"Otsu score: {otsu_score}")

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

    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        continue

    approx = cv.approxPolyDP(contour, 0.02*perimeter, True)

    vertices = len(approx)
    circularity = 4 * math.pi * area / (perimeter*perimeter)

    shape = "none"
    if vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        x,y,w,h = cv.boundingRect(approx)
        ratio = w / float(h)

        if 0.90 <= ratio <= 1.10:
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

    cv.drawContours(output,[approx],-1,(0,255,0),2)
    cv.putText(
        output,
        shape,
        (x, max(y - 10, 20)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    moments = cv.moments(contour)
    if moments["m00"] == 0: # aria
        continue

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    cv.circle(
        output,
        (cx,cy),
        5,
        (255,0,0),
        -1
    )
    # centroid coords
    cv.putText(
        output,
        shape,
        (cx+10,max(cy-20,20)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,0,255),
        2
    )
    cv.putText(
        output,
        f"({cx},{cy})",
        (cx+10,cy),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,0,255),
        2
    )

cv.imshow('thresh', thresh)
cv.imshow('output', output)
cv.waitKey(0)
cv.destroyAllWindows()

