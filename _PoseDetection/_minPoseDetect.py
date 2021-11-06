# Below is code used to track the Position of a user through usage of the
# webcam. The specific iteration changes the color of the nose just to
# showcase a little bit of customization that can be done.
# please refer to: https://google.github.io/mediapipe/solutions/pose
# for more details and documentation

# Imports
import cv2
import mediapipe as mp
import time

# OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
# Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# To access a video: cap = cv2.VideoCapture('/pathtovideo.mp4')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Calling Pose Detection Modules from Mediapipe
mpDraw = mp.solutions.drawing_utils # Draw lines between landmarks
mpPose = mp.solutions.pose
pose = mpPose.Pose() # write needed parameters, ctrl + click

# frame rate Initialization
pTime = 0

while True:
    # Read Image Capture
    success, img = cap.read()

    # convert image since mediapipe only uses RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # Print Landmark Locations
    if results.pose_landmarks:
        # Draw pose Connections
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Loop to get info of Landmark Information
        for id, lm in enumerate(results.pose_landmarks.landmark):

            # get image shape
            h, w, c = img.shape
            # convert decimal representations to integers
            cx, cy = int(lm.x * w), int(lm.y *h)
            # print(id, cx, cy) # information on location specific data

            # OPTIONAL: Make Nose Blue
            if id == 0:
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

    # Frame Rate
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    # Output fps to image
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # Show Live Image
    cv2.imshow("image", img)
    cv2.waitKey(1)

# clean capture
cap.release()