import cv2 as cv


def frameModifier(frame):
    frame = cv.flip(frame, 1)
    frame = cv.blur(frame, (3, 3))
    frame = cv.Sobel(frame, cv.CV_8U, 1, 0, 3, 7, 128)
    return frame


cap = cv.VideoCapture(0)

if not cap.isOpened:
    print('Can\'t open the camera! :(')
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print('Can\'t receive frame; ending the capture.')
        break

    cv.imshow('frame', frame)

    frame = frameModifier(frame)
    cv.imshow('frame but modified', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
