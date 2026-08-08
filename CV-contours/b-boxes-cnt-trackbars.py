# TO DO: implement it
import cv2 as cv

img = cv.imread('ratunel.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def nothing(x):
    pass

cv.namedWindow('ratunel')
cv.createTrackbar('area','ratunel',0,5000,nothing)

# thresh
_, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
# contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cnt = img.copy()

output = 0
#draw contours
cv.drawContours(
    cnt,
    contours,
    -1,
    (0,255,0),
    2
)

mode = "main"
while True:
    k = cv.getTrackbarPos('area','ratunel')

    boxes = img.copy()
    obj = 0
    for contour in contours:
        # filter
        area = cv.contourArea(contour)

        if area < k:
            continue
        x, y, w, h = cv.boundingRect(contour)

        cv.rectangle(
            boxes,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv.putText(
            boxes,
            f"Area: {int(area)}",
            (x, max(y-8,15)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255,0,0),
            1
        )

        obj += 1

    cv.putText(
        boxes,
        f"Min area: {k} | Obj: {obj}",
        (10,20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        mode = "contours"
    elif key == ord('x'):
        mode = "b-boxes"
    elif key == ord('o'):
        mode = "main"

    if mode == "contours":
        output = cnt
    elif mode == "b-boxes":
        output = boxes
    elif mode == "main":
        output = img

    cv.imshow('ratunel', output)


cv.destroyAllWindows()