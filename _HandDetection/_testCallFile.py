import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

# Frame Rate
pTime = 0

# OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
# Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Call Hand Detector Class
detector = htm.handDetector()

while True:
    # Read Image Capture
    success, img = cap.read()

    # Call detector to find hands
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)

    # Output information of specific Landmark
    #if len(lmlist) != 0:
        #print(lmlist[8])

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Output fps to image
    #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
    #            3, (255, 0, 255), 3)

    # Show Live Image
    cv2.imshow("Image", img)
    cv2.waitKey(1)