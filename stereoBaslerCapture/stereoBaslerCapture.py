import cv2
import numpy as np
from pypylon import pylon
import os

maxCamerasToUse = 2

# Get the transport layer factory.
tlFactory = pylon.TlFactory.GetInstance()

# Get all attached devices and exit application if no device is found.
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RUNTIME_EXCEPTION("No camera present.")

# Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

l = cameras.GetSize()

# Create and attach all Pylon Devices.
for i, cam in enumerate(cameras):
    cam.Attach(tlFactory.CreateDevice(devices[i]))
    

    # Print the model name of the camera.
    print("Using device ", cam.GetDeviceInfo().GetModelName())

    # Set inital camera properties
    cam.Open()
    cam.GetNodeMap().GetNode("ExposureAuto").SetValue("Off")
    cam.GetNodeMap().GetNode("ExposureTimeRaw").SetValue(40000)
    cam.GetNodeMap().GetNode("GainAuto").SetValue("Off")
    cam.GetNodeMap().GetNode("GainRaw").SetValue(0)
    cam.GetNodeMap().GetNode("PixelFormat").SetValue("Mono8")
    cam.GetNodeMap().GetNode("Width").SetValue(2448)
    cam.GetNodeMap().GetNode("Height").SetValue(2048)
    cam.Close()

cameras.StartGrabbing()
cv2.namedWindow('stereoBaslerCameras', cv2.WINDOW_NORMAL)

converter = pylon.ImageFormatConverter()
#converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputPixelFormat = pylon.PixelType_Mono8
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

firstImageReceivedR = False
firstImageReceivedL = False

while cameras.IsGrabbing():
    grabResult = cameras.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        cameraContextValue = grabResult.GetCameraContext()
        # Access the image data.
        img = grabResult.Array
        # Convert to opencv image
        image = converter.Convert(grabResult)
        print("Camera ", cameraContextValue, ": ", cameras[cameraContextValue].GetDeviceInfo().GetModelName())
        frame = image.GetArray()
        if (not firstImageReceivedR):
            frame_right = np.zeros(shape=frame.shape,dtype=np.uint8)
        if (not firstImageReceivedL):
            frame_left = np.zeros(shape=frame.shape,dtype=np.uint8)
        if cameraContextValue == 0:
            frame_left = frame
            firstImageReceivedL = True
        elif cameraContextValue == 1:
            frame_right = frame
            firstImageReceivedR = True
    
        frame_joint = np.concatenate((frame_left,frame_right), axis=1)
        # Display image
        cv2.imshow('stereoBaslerCameras', frame_joint)
        # Close on ESC
        k = cv2.waitKey(1)
        if k == 27:
            break
    

cameras.StopGrabbing()
cv2.destroyAllWindows()
