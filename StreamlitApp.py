import cv2

camera = cv2.VideoCapture(1)  # Try changing this to 1, 2, or other values

if not camera.isOpened():
    print("Error: Webcam is not accessible.")
else:
    print("Webcam is working fine!")

camera.release()
