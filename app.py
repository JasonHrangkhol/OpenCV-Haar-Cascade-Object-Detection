import cv2 as cv
import time

# Load the Haar Cascade
face_cascade_path = "Cascades\haarcascade_frontalface_default.xml"
eye_cascade_path = "Cascades\haarcascade_eye.xml"
mouth_cascade_path = "Cascades\haarcascade_smile.xml"

#Read the Haar Cascade
face_cascade = cv.CascadeClassifier(face_cascade_path)
eye_cascade = cv.CascadeClassifier(eye_cascade_path)
mouth_cascade = cv.CascadeClassifier(mouth_cascade_path)

#Initialize the video stream and allow the camera sensor to warm up
capture = cv.VideoCapture(0)

time.sleep(2.0)

#loop over the frames from the video stream
while True:

    #grab the frame from the video
    ret,frame = capture.read()

    #Convert to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Perform face detection using the appropriate haar cascade
    faces = face_cascade.detectMultiScale(gray_frame,1.05,5)

    #loop over the face bounding boxes
    for face in faces:

        #extract the boundary of each face box 
        fx,fy,fw,fh = face
        
        #extract the face from frame
        face_frame = gray_frame[fy:fy+fh, fx:fx+fw]

        #apply eyes detection to the face
        eyes = eye_cascade.detectMultiScale(face_frame,1.1,10)

        #apply mouth detection to the face
        mouths = mouth_cascade.detectMultiScale(face_frame,1.1,10)

        #loop over the eye bounding boxes
        for eye in eyes:

            #extract the boundary for each eye box  
            ex,ey,ew,eh = eye

            #draw the eye bounding box
            cv.rectangle(frame,(fx+ex,fy+ey),(fx+ex+ew,fy+ey+eh),(255,0,0),2)
        
        #loop over the smile bounding boxes
        for mouth in mouths:
            
            #extract the boundary for each mouth box
            mx,my,mw,mh = mouth

            #draw the mouth bounding box
            cv.rectangle(frame,(fx+mx,fy+my),(fx+mx+mw,fy+my+mh),(0,255,0),2)

        #draw the face bounding box
        cv.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)

    #display the resulting frame
    cv.imshow("Webcam Feed",frame)

    #if the 'q is pressed, close the webcam feed
    if cv.waitKey(1)==ord('q'):
        break    

#when everything is done release the capture
capture.release()
cv.destroyAllWindows()