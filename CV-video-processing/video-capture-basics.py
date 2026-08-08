import cv2 as cv

cap = cv.VideoCapture(0) # default camera

if not cap.isOpened():
    print("error")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. exiting")
        exit()

    cv.imshow('Camera', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()