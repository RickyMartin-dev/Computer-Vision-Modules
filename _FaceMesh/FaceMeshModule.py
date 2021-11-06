# Creating a Hand Tracking Module to call upon for other Projects
# Refer to : https://google.github.io/mediapipe/solutions/face_mesh
# For more Information

import cv2
import mediapipe as mp
import time

# Class to recognize Face Mesh
class FaceMeshDetect():

    # Initializations
    def __init__(self, static=False, max_faces=2, min_detect=0.5, min_track=0.5):
        self.static = static
        self.max_faces = max_faces
        self.min_detect = min_detect
        self.min_track = min_track

        # Face Tracking Modules
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static, self.max_faces, self.min_detect, self.min_track)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    # Find Mesh
    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        # Face Landmarks
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # Draw Connections
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                          self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
            faces.append(face)
        return img, faces

def main():
    # Frame Rate
    pTime = 0

    # OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
    # Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # To access a video: cap = cv2.VideoCapture('/pathtovideo.mp4')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Call Face Mesh Class
    detector = FaceMeshDetect()
    while True:
        # Read Image Capture
        success, img = cap.read()

        # Call detector to find Mesh
        img, faces = detector.findFaceMesh(img)

        # Output Mesh Variables
        #if len(faces) != 0:
            #print(len(faces))

        # Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Output fps to image
        cv2.putText(img, f'{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 2)

        # Show Live Image
        cv2.imshow("image", img)
        cv2.waitKey(1)


# Copy def main() to run in another file
if __name__ == "__main__":
    main()