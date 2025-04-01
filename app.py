import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize Streamlit
st.title("Finger Drawing App ✋🎨")
st.write("Raise your thumb and move your index finger to draw.")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Color selection
colors = {"Green": (0, 255, 0), "Red": (0, 0, 255), "Blue": (255, 0, 0), "Yellow": (0, 255, 255)}
selected_color = st.selectbox("Choose a color:", list(colors.keys()))

# Clear button
if "canvas" not in st.session_state:
    st.session_state["canvas"] = None
if "last_point" not in st.session_state:
    st.session_state["last_point"] = None
if st.button("Clear Canvas"):
    st.session_state["canvas"] = None
    st.session_state["last_point"] = None

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Camera not accessible.")
else:
    frame_placeholder = st.empty()

# Initialize canvas
canvas = None

# Video stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Video capture failed!")
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    h, w, _ = frame.shape

    # Initialize drawing canvas
    if st.session_state["canvas"] is None:
        st.session_state["canvas"] = np.zeros_like(frame)

    # Convert frame to RGB (for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip position
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # Check if thumb is up (to start drawing)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            is_thumb_up = thumb_tip.y < thumb_ip.y

            if is_thumb_up:
                if st.session_state["last_point"] is None:
                    st.session_state["last_point"] = (x, y)
                else:
                    # Draw a smooth line from last point to current point
                    cv2.line(st.session_state["canvas"], 
                             st.session_state["last_point"], 
                             (x, y), 
                             colors[selected_color], 5)
                st.session_state["last_point"] = (x, y)
            else:
                st.session_state["last_point"] = None

    # Merge canvas with video frame
    frame = cv2.addWeighted(frame, 1, st.session_state["canvas"], 1, 0)

    # Display updated frame
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)

cap.release()
cv2.destroyAllWindows()
