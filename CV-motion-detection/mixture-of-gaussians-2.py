import cv2 as cv

cap = cv.VideoCapture(0)
window = 'Camera'

if not cap.isOpened():
    raise RuntimeError("Camera is not working")

def nothing(x):
    pass

cv.namedWindow(window)
cv.createTrackbar('thresh', window, 200, 255, nothing)
cv.createTrackbar('min_area', window, 500, 5000, nothing)

kernel = cv.getStructuringElement(
    cv.MORPH_RECT,
    (5, 5)
)

bg = cv.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)
while True:
    ret, crt_frame = cap.read()

    if not ret:
        break

    output = crt_frame.copy()

    mask = bg.apply(crt_frame)

    thresh = cv.getTrackbarPos('thresh', window)
    k = cv.getTrackbarPos('min_area', window)

    _, mask = cv.threshold(
        mask,
        thresh,
        255,
        cv.THRESH_BINARY
    )
    mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        kernel
    )

    mask = cv.dilate(
        mask,
        kernel,
        iterations=2
    )

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    motion = False
    for contour in contours:
        area = cv.contourArea(contour)

        if area < k:
            continue

        motion = True

        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        cv.putText(
            output,
            f"Area: {int(area)}",
            (x, max(y - 20, 20)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            2
        )
    status = "Motion Detected!" if motion else "No motion detected!"

    cv.putText(
        output,
        status,
        (10, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255) if motion else (0, 255, 255),
        2

    )
    cv.imshow(window, output)
    cv.imshow('Mask', mask)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()