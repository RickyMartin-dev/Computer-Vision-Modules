# Creating a Hand Tracking Module to call upon for other Projects
# Refer to : https://google.github.io/mediapipe/solutions/face_detection
# For more Information

import cv2
import mediapipe as mp
import time

# Class to recognize face
class FaceDetector():
    # Initializations
    def __init__(self, mindetectCon=0.5):

        self.mindetectCon = mindetectCon

        # Hand Tracking Modules
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.mindetectCon)

    # Detect Faces
    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(results)
        bboxs = []

        # Get Position Information
        if self.results.detections:
            for id, detect in enumerate(self.results.detections):
                bboxC = detect.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detect.score])

                # Calls to Draw Box
                if draw:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detect.score[0] * 100)} %',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # Create Rectangle
        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        # Top Left, x,y
        cv2.line(img, (x,y), (x+l,y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right, x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # bottom Left, x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # bottom Right, x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img

def main():
    # Frame Rate
    pTime = 0

    # OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
    # Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # To access a video: cap = cv2.VideoCapture('/pathtovideo.mp4')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Call Hand Detector Class
    detector = FaceDetector()

    while True:
        # Read Image Capture
        success, img = cap.read()

        # Call detector to find hands
        img, bboxs = detector.findFaces(img)

        # Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Output fps to image
        cv2.putText(img, f'{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 2)

        # Show Live Image
        cv2.imshow("Image", img)
        cv2.waitKey(1)


# Copy def main() to run in another file
if __name__ == "__main__":
    main()