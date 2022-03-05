import cv2 as cv
import time


face_cascade_path = "Cascades\haarcascade_frontalface_default.xml"
eye_cascade_path = "Cascades\haarcascade_eye.xml"
mouth_cascade_path = "Cascades\haarcascade_smile.xml"

face_cascade = cv.CascadeClassifier(face_cascade_path)
eye_cascade = cv.CascadeClassifier(eye_cascade_path)
mouth_cascade = cv.CascadeClassifier(mouth_cascade_path)

capture = cv.VideoCapture(0)
time.sleep(2.0)

while True:

    ret,frame = capture.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame,1.05,5)

    for face in faces:

        fx,fy,fw,fh = face
        
        face_frame = gray_frame[fy:fy+fh, fx:fx+fw]

        eyes = eye_cascade.detectMultiScale(face_frame,1.1,10)

        mouths = mouth_cascade.detectMultiScale(face_frame,1.1,10)

        for eye in eyes:

            ex,ey,ew,eh = eye
            cv.rectangle(frame,(fx+ex,fy+ey),(fx+ex+ew,fy+ey+eh),(255,0,0),2)
        
        for mouth in mouths:

            mx,my,mw,mh = mouth
            cv.rectangle(frame,(fx+mx,fy+my),(fx+mx+mw,fy+my+mh),(0,255,0),2)
    
        cv.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)

    cv.imshow("Webcam Feed",frame)

    if cv.waitKey(1)==ord('q'):
        break    

capture.release()
cv.destroyAllWindows()