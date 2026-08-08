import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError('Camera is not working')

ret, old_frame = cap.read()
if not ret:
    cap.release()
    exit()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0= cv.goodFeaturesToTrack(
    old_gray,
    mask=None,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Lucas Kanade algorithm implementation
lk_params = dict(
    winSize=(15,15),
    maxLevel=2,
    criteria=(
        cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

mask = np.zeros_like(old_frame)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, status, error = cv.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    good_new = p1[status == 1]
    good_old = p0[status == 1]

    # draw flow movement
    for new, old in zip(good_new, good_old):
        new_x, new_y = new.ravel()
        old_x, old_y = old.ravel()

        new_point = (int(new_x), int(new_y))
        old_point = (int(old_x), int(old_y))

        mask = cv.line(
            mask,
            new_point,
            old_point,
            (0,255,0),
            2,
            cv.LINE_AA
        )

        frame = cv.circle(
            frame,
            new_point,
            5,
            (0,0,255),
            -1
        )

    output = cv.add(frame,mask)
    cv.imshow('Optical flow', output)

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()