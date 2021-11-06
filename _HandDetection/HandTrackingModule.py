# Creating a Hand Tracking Module to call upon for other Projects
# Refer to : https://google.github.io/mediapipe/solutions/hands
# For more Information

import cv2
import mediapipe as mp
import time

# Class to recognize hands
class handDetector():

    # Initializations
    def __init__(self, mode=False, maxhands=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectCon = detectCon
        self.trackCon = trackCon

        # Hand Tracking Modules
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands,
                                        self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # Drawing Module

    # Find Hands
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Draw Connections
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                           self.mpHands.HAND_CONNECTIONS)
        return img

    # Find Position of Landmarks
    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # Find Position Information
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                # Draws ET Finger if you Want
                #if draw:
                #    if id == 8:
                #        cv2.circle(img, (cx,cy), 10, (255, 0, 255), cv2.FILLED)
        return lmlist

def main():
    # Frame Rate
    pTime = 0

    # OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
    # Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # To access a video: cap = cv2.VideoCapture('/pathtovideo.mp4')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Call Hand Detector Class
    detector = handDetector()

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
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 255), 3)

        # Show Live Image
        cv2.imshow("Image", img)
        cv2.waitKey(1)


# Copy def main() to run in another file
if __name__ == "__main__":
    main()