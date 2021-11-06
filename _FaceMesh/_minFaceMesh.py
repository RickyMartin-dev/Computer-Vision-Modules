# Below is code used to track the hands of a user through usage of the
# webcam. The specific iteration increases the size of thumbs just to
# showcase a little bit of customization that can be done.
# please refer to: https://google.github.io/mediapipe/solutions/face_mesh
# for more details and documentation

# Imports
import cv2
import mediapipe as mp
import time

# OpenCV video capture, Mac/Linux use: cap = cv2.VideoCapture(0)
# Windows use cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# To access a video: cap = cv2.VideoCapture('/pathtovideo.m4')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Calling Face Mesh Modules from Mediapipe
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# frame rate Initialization
pTime = 0

while True:
    # Read Image Capture
    success, img = cap.read()

    # convert image since mediapipe only uses RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    # Check Hand Landmarks
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            # Loop to get info of Landmark Information
            for id, lm in enumerate(faceLms.landmark):
                # Get image Shape
                h, w, c = img.shape
                x, y = int(lm.x*w), int(lm.y*h)
                #print(id, x, y)

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