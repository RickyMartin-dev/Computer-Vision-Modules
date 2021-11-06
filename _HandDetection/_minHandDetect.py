# Below is code used to track the hands of a user through usage of the
# webcam. The specific iteration increases the size of thumbs just to
# showcase a little bit of customization that can be done.
# please refer to: https://google.github.io/mediapipe/solutions/hands
# for more details and documentation

# Imports
import cv2
import mediapipe as mp
import time

# OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
# Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# To access a video: cap = cv2.VideoCapture('/pathtovideo.m4')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Calling Hand Detection Modules from Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands() # write needed parameters, ctrl + click
mpDraw = mp.solutions.drawing_utils # Draw lines between landmarks

# frame rate Initialization
pTime = 0

while True:
    # Read Image Capture
    success, img = cap.read()

    # convert image since mediapipe only uses RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Check Hand Landmarks
    if results.multi_hand_landmarks:
        for handLmrks in results.multi_hand_landmarks:

            # Loop to get info of Landmark Information
            for id, lm in enumerate(handLmrks.landmark):

                # get image shape
                h, w, c = img.shape
                # convert decimal representations to integers
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy) # information on location specific data

                # OPTIONAL: Make Index Finger Tips Big, ET Phone home
                if id == 8:
                    cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)

            # Draw hand Connections
            mpDraw.draw_landmarks(img, handLmrks, mpHands.HAND_CONNECTIONS)

    # Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Output fps to image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                3, (255,0,255), 3)

    # Show Live Image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# clean capture
cap.release()