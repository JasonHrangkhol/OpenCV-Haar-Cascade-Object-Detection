import cv2 as cv
import time

capture = cv.VideoCapture(0)
time.sleep(2.0)

while True:

    ret,frame = capture.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
   
    cv.imshow("Webcam Feed",frame)

    if cv.waitKey(1)==ord('q'):
        break    

capture.release()
cv.destroyAllWindows()