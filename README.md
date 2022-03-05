Object Detection using Python 

This project on Object Detection has been done with the implementation of Haar Cascade in Python scripts.
It includes:

Face Detection
Eye Detection
Mouth Detection

Haar Cascades

Haar cascades are notoriously prone to false-positives — the Viola-Jones algorithm can easily report a face in an image when no face is present.

There will be times when we can detect all the faces in an image. There will be other times when 

(1) regions of an image are falsely classified as faces, and/or 
(2) faces are missed entirely.

OpenCV’s face detection Haar cascades tend to be the most accurate. 

We will apply three Haar cascades to a real-time video stream. These Haar cascades reside in the cascades directory and include:

haarcascade_frontalface_default.xml: Detects faces
haarcascade_eye.xml: Detects the left and right eyes on the face
haarcascade_smile.xml: While the filename suggests that this model is a “smile detector,” it actually detects the presence of the “mouth” on a face

Our opencv_haar_cascades.py script will load these three Haar cascades from disk and apply them to a video stream, all in real-time.
