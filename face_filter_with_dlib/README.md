# Face Filter with Dlib

This Python project applies a filter to a detected face in real-time using a webcam. It leverages the `dlib` library for facial landmark detection and OpenCV for image processing.

## Features

- Real-time face detection using `dlib`
- Applies a custom filter (e.g., glasses or other overlay) to the detected face
- Dynamically resizes and positions the filter based on eye landmarks
- Uses a webcam as the video input

## Prerequisites

To run this project, make sure you have the following installed:

- Python 3.7 or higher
- OpenCV (`cv2`)
- NumPy
- dlib

You will also need:

1. A webcam for real-time input.
2. The pre-trained facial landmark model: [`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
3. A filter image with an alpha channel (e.g., `filter.png`).

