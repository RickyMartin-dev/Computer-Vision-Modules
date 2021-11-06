# Below is code used to track the hands of a user through usage of the
# webcam. The specific iteration increases the size of thumbs just to
# showcase a little bit of customization that can be done.
# please refer to: https://google.github.io/mediapipe/solutions/face_detection
# for more details and documentation

# Imports
import cv2
import mediapipe as mp
import time

# OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
# Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# To access a video: cap = cv2.VideoCapture('/pathtovideo.m4')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# frame rate Initialization
pTime = 0

# Calling Face Detection Modules from Mediapipe
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    # Read Image Capture
    success, img = cap.read()

    # convert image since mediapipe only uses RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    # Check Face LandMarks
    if results.detections:

        # Loop to get info of Landmark Information
        for id, detect in enumerate(results.detections):
            #print(id, detect)
            mpDraw.draw_detection(img, detect)
            bboxC = detect.location_data.relative_bounding_box

            # Get Image Shape
            ih, iw, ic = img.shape
            # Get Box info Shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            # Create Box Around Face
            cv2.rectangle(img, bbox, (255, 0, 0), 2)
            # Output face detection Confidence
            cv2.putText(img, f'{int(detect.score[0] * 100)} %',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Output fps to image
    cv2.putText(img, f'{int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 2)

    # Show Live Image
    cv2.imshow("image", img)
    cv2.waitKey(1)

# clean capture
cap.release()