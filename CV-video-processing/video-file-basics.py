import cv2 as cv

cap = cv.VideoCapture('video.mov')

if not cap.isOpened():
    print("error")
    exit()



while True:
    ret, frame = cap.read()
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_cnt = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if not ret:
        break
    cv.putText(
        frame,
        "FPS: " + str(fps),
        (10,20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        2
    )
    cv.imshow('Video', frame)

    if cv.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
