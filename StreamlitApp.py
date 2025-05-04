import cv2
import streamlit as st
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
camera = cv2.VideoCapture(0)

# Streamlit page title
st.title("Finger Drawing App")

# Color selection for drawing
color = st.selectbox("Select Color", ["green", "red", "blue", "yellow"], key="color")

# Initialize the drawing canvas
canvas = None
last_point = None
colors = {'green': (0, 255, 0), 'red': (0, 0, 255), 'blue': (255, 0, 0), 'yellow': (0, 255, 255)}
line_thickness = 3

# Function to check if the thumb is raised
def is_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    return thumb_tip.y < thumb_ip.y

# Function to check if the index finger is pointing
def is_index_pointing(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    return index_tip.y < index_pip.y

# Open Streamlit's image window
FRAME_WINDOW = st.image([])

# Run webcam feed
run = st.checkbox('Run Webcam Feed')

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Error: Couldn't access the webcam.")
        break
    
    # Convert the frame to RGB (MediaPipe expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Initialize canvas if not already initialized
    if canvas is None:
        canvas = np.zeros_like(frame)

    # If hands are detected, process the landmarks and draw
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip coordinates
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_finger.x * w), int(index_finger.y * h)
            
            # Check if thumb is up and index finger is pointing
            if is_thumb_up(hand_landmarks) and is_index_pointing(hand_landmarks):
                if last_point is None:
                    last_point = (x, y)
                else:
                    cv2.line(canvas, last_point, (x, y), colors[color], line_thickness)
                last_point = (x, y)
            else:
                last_point = None

    # Combine drawing with the original frame
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)
    
    # Show the frame in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

# Close the webcam when done
camera.release()
