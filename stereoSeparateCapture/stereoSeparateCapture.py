import cv2
import numpy as np

camL = cv2.VideoCapture(0)
camR = cv2.VideoCapture(1)
cv2.namedWindow("stereoCamera")

while True:
    retL, frame_left = camL.read()
    retR, frame_right = camR.read()

    frame_joint = np.concatenate((frame_left,frame_right), axis=1)

    cv2.imshow("stereoCamera", frame_left)

    if not retL:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break