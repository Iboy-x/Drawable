import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam is not accessible.")
else:
    print("Webcam is working fine!")

camera.release()
