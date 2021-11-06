# Computer-Vision-Modules

The modules presented within this repository are simplistic Computer Vision Modules to build other projects atop of. Simply pull this repo and import these modules into other projects to be used. Below are outlines of what this repository contains as well as the documentation needed to further increase learning.

In order to utilize these modules, the OpenCV and MediaPipe Libraries are needed.
OpenCV Homepage: https://opencv.org/
MediaPipe Homepage: https://google.github.io/mediapipe/

All the modules below occur in real-time and utilize the users webcam, however with simple changes to the media input, you could very easily allow this module to work with real time video.


## Hand Detection
Please refer to: https://google.github.io/mediapipe/solutions/hands for more information.
This module allows for the perception of the shape and motion of hands. 21 landmarks are given to represent the major parts of the users hands, for example: landmark 8 (tip of index finger). 

Example use-cases: Sign Language detector, augmented reality hand tracking, hand gesture controls, robotic hand control.


## Face Detection
Please refer to: https://google.github.io/mediapipe/solutions/face_detection for more information.
This module allows for the perception of faces. 6 landmarks are given to represent the major parts of the users face.

Example use-cases: Camera Focus, Personel Recognition, face tracking security camera, eye tracker.

## Face Mesh
Please refer to: https://google.github.io/mediapipe/solutions/face_mesh for more information.
This module allows for the perception of the shape of faces. 468 landmarks are given to represent the complex shape of the face.

Example use-cases: Social media face filter, virtual face augmentation, deep fake technologies.

## Pose Tetection
Please refer to: https://google.github.io/mediapipe/solutions/pose for more information.
This module allows for the perception of the shape and motion of hands. 33 landmarks are given to represent the major parts of the users body. 

Example use-cases: Augmented Reality, Motion Capture, AI Personal Trainer, Remote Physical Therapy.
