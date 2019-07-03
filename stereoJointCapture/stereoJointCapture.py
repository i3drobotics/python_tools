import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cv2.namedWindow("stereoCamera")

img_counter = 0

while True:
    ret, frame = cam.read()

    frame_left = frame.copy()
    frame_left = cv2.flip(frame_left[:, :, 1],0)
    frame_right = frame.copy()
    frame_right = cv2.flip(frame_right[:, :, 2],0)

    frame_joint = np.concatenate((frame_left,frame_right), axis=1)

    cv2.imshow("stereoCamera", frame_joint)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break