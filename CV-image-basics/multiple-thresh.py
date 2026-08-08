import cv2 as cv

img = cv.imread('ratunel.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
def nothing(x):
    pass
cv.namedWindow('ratunel gallery')
cv.createTrackbar('threshold', 'ratunel gallery', 0,255,nothing)
cv.createTrackbar('kernel', 'ratunel gallery', 1,21,nothing)
cv.createTrackbar('iterations','ratunel gallery', 1,10,nothing)
mode = "original"

while True:
    t1 = cv.getTrackbarPos('threshold', 'ratunel gallery')

    _, thresh = cv.threshold(
        gray,
        t1,
        255,
        cv.THRESH_BINARY
    )

    _, thresh_inv = cv.threshold(
        gray,
        t1,
        255,
        cv.THRESH_BINARY_INV
    )

    _, truncate_tr = cv.threshold(
        gray,
        t1,
        255,  # truncate
        cv.THRESH_TRUNC
    )
    _, to_zero = cv.threshold(
        gray,
        t1,
        255,
        cv.THRESH_TOZERO
    )
    adaptive = cv.adaptiveThreshold(
        gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11,
        2
    )

    _, otsu = cv.threshold(
        gray,
        0,
        255,
        cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    k = cv.getTrackbarPos('kernel', 'ratunel gallery')
    iter = cv.getTrackbarPos('iterations', 'ratunel gallery')
    if k <1:
        k = 1
    if k%2==0:
        k+=1
    # kernel = np.ones((5,5), np.uint8)
    kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        (k,k)
    )
    erosion = cv.erode(
        thresh,
        kernel,
        iterations=iter
    )

    dilation = cv.dilate(
        thresh,
        kernel,
        iterations=iter
    )
    # open = erode + dilate
    opening = cv.morphologyEx(
        thresh,
        cv.MORPH_OPEN,
        kernel,
        iterations=iter
    )

    closing = cv.morphologyEx(
        thresh,
        cv.MORPH_CLOSE,
        kernel,
        iterations=iter
    )

    key = cv.waitKey(1) & 0xFF
    output = None

    if key == ord('q'):
        break
    elif key == ord('w'):
        mode = "original"
    elif key == ord('e'):
        mode = "binary"
    elif key == ord('r'):
        mode = "binary_inv"
    elif key == ord('t'):
        mode = "truncate"
    elif key == ord('y'):
        mode = "zero"
    elif key == ord('u'):
        mode = "adaptive"
    elif key == ord('i'):
        mode = "otsu"
    elif key == ord('a'):
        mode = "erode"
    elif key == ord('s'):
        mode = "dilation"
    elif key == ord('d'):
        mode = "opening"
    elif key == ord('f'):
        mode = "closing"

    if mode == "binary":
        output = thresh.copy()
    elif mode == "binary_inv":
        output = thresh_inv.copy()
    elif mode == "truncate":
        output = truncate_tr.copy()
    elif mode == "zero":
        output = to_zero.copy()
    elif mode == "adaptive":
        output = adaptive.copy()
    elif mode == "otsu":
        output = otsu.copy()
    elif mode == "erode":
        output = erosion.copy()
    elif mode == "dilation":
        output = dilation.copy()
    elif mode == "opening":
        output = opening.copy()
    elif mode == "closing":
        output = closing.copy()
    else:
        output = img.copy()
    # print(f"current mode: {mode}")
    cv.putText(
        output,
        f"Mode: {mode} | Threshold: {t1}",
        (10,20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255,255,255),
        1
    )

    cv.imshow('ratunel gallery', output)

cv.destroyAllWindows()