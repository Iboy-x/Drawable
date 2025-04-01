import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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

# WebRTC Video Processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.canvas = None
        self.last_point = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror image
        h, w, _ = img.shape

        # Initialize canvas
        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # Convert frame to RGB (for MediaPipe)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip position
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)

                # Check if thumb is up (to start drawing)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                is_thumb_up = thumb_tip.y < thumb_ip.y

                if is_thumb_up:
                    if self.last_point is None:
                        self.last_point = (x, y)
                    else:
                        # Draw a smooth line from last point to current point
                        cv2.line(self.canvas, self.last_point, (x, y), colors[selected_color], 5)
                    self.last_point = (x, y)
                else:
                    self.last_point = None

        # Merge canvas with video frame
        img = cv2.addWeighted(img, 1, self.canvas, 1, 0)

        return img

# Run WebRTC Streamer
webrtc_streamer(key="draw", video_transformer_factory=VideoTransformer)
