# Creating a Pose Tracking Module to call upon for other Projects
# Refer to : https://google.github.io/mediapipe/solutions/pose
# For more Information

import cv2
import mediapipe as mp
import time

# Class to recognize Pose
class poseDetector():

    # Initializations
    def __init__(self, mode=False, complex=1, smoothLand=True, upBody=False,
                 smooth=True, detectCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complex = complex
        self.smoothLand = smoothLand
        self.upBody = upBody
        self.smooth = smooth
        self.detectCon = detectCon
        self.trackCon = trackCon

        # Pose Tracking Modules
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complex, self. smoothLand,
                                     self.upBody, self.smooth, self.detectCon, self.trackCon)

    # Find and draw Positions
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # Draw Connections
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    # Get position specific data
    def findPosition(self, img, draw=True):
        lmList = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y *h)
            lmList.append([id, cx, cy])
            # Draws Blue Nose if you Want
            #if draw:
            #    if id == 0:
            #        cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList

def main():
    # Frame Rate
    pTime = 0

    # OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
    # Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # To access a video: cap = cv2.VideoCapture('/pathtovideo.mp4')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Call Pose Detector Class
    detector = poseDetector()

    while True:
        # Read Image Capture
        success, img = cap.read()

        # Call detector to find Position
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        # Output information of specific Landmark
        #if len(lmList) != 0:
        #    #print(lmList[0])
        #    cv2.circle(img, (lmList[0][1], lmList[0][2]), 5, (255, 0, 0), cv2.FILLED)

        # Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Output fps to image
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        # Show Live Image
        cv2.imshow("image", img)
        cv2.waitKey(1)


# Copy def main() to run in another file
if __name__ == "__main__":
    main()