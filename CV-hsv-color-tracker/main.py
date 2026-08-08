import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error opening cam")
    exit()
def nothing(x):
    pass
cv.namedWindow('Camera')
cv.createTrackbar('H min', 'Camera', 0, 179, nothing)
cv.createTrackbar('H max', 'Camera', 179,179, nothing)
cv.createTrackbar('S min', 'Camera', 0,255, nothing)
cv.createTrackbar('S max', 'Camera', 255,255, nothing)
cv.createTrackbar('V min', 'Camera', 0,255, nothing)
cv.createTrackbar('V max', 'Camera', 255,255, nothing)

kernel = cv.getStructuringElement(
    cv.MORPH_RECT,
    (5, 5),
)
MEM_MAX = 100 # points memory limit
points = []
while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos('H min', 'Camera')
    h_max = cv.getTrackbarPos('H max', 'Camera')
    s_min = cv.getTrackbarPos('S min', 'Camera')
    s_max = cv.getTrackbarPos('S max', 'Camera')
    v_min = cv.getTrackbarPos('V min', 'Camera')
    v_max = cv.getTrackbarPos('V max', 'Camera')

    mask = cv.inRange(
        hsv,
        (h_min,s_min,v_min),
        (h_max,s_max,v_max)
    )


    opening = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        kernel
    )

    contours, hierarchy = cv.findContours(
        opening,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )
    output = frame.copy()

    if contours:
        largest = max(contours, key=cv.contourArea)
        area = cv.contourArea(largest)

        if area > 500:
            x,y,w,h = cv.boundingRect(largest)

            cv.rectangle(
                output,
                (x,y),
                (x+w,y+h),
                (0,255,255),
                2
            )
            moments = cv.moments(largest)
            if moments["m00"] != 0:

                cx = int(moments["m10"]/moments["m00"])
                cy = int(moments["m01"]/moments["m00"])

                points.append((cx,cy))
                if len(points) > MEM_MAX:
                    points.pop(0)

                for i in range(1, len(points)):
                    cv.line(
                        output,
                        points[i-1],
                        points[i],
                        (0,255,255),
                        2,
                    )
                cv.circle(
                    output,
                    (cx,cy),
                    5,
                    (255,0,0),
                    -1
                )
                cv.putText(
                    output,
                    f"Center: ({cx},{cy}) | Area: {int(area)}",
                    (10,30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255),
                    2
                )
    cv.imshow('Camera', output)
    cv.imshow('Mask', opening)


    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        points.clear()

cap.release()
cv.destroyAllWindows()