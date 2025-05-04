import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Finger Drawing Live", layout="wide")

class FingerDrawing(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.last_point = None
        self.canvas = None
        self.current_color = (0, 255, 0)  # default green
        self.line_thickness = 3

    def is_thumb_up(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        return thumb_tip.y < thumb_ip.y

    def is_index_pointing(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        return index_tip.y < index_pip.y

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                index = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cx, cy = int(index.x * w), int(index.y * h)

                if self.is_thumb_up(hand_landmarks) and self.is_index_pointing(hand_landmarks):
                    if self.last_point is None:
                        self.last_point = (cx, cy)
                    else:
                        cv2.line(self.canvas, self.last_point, (cx, cy), self.current_color, self.line_thickness)
                        self.last_point = (cx, cy)
                else:
                    self.last_point = None

        combo = cv2.addWeighted(img, 1, self.canvas, 1, 0)
        return combo

# Sidebar options
st.sidebar.title("🎨 Drawing Options")
color_choice = st.sidebar.selectbox("Pick a color", ["Green", "Red", "Blue", "Yellow"])
color_map = {
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
}
selected_color = color_map[color_choice]

# Inject color into our transformer
drawing_transformer = FingerDrawing()
drawing_transformer.current_color = selected_color

# WebRTC Streamer
webrtc_streamer(key="drawing", video_transformer_factory=lambda: drawing_transformer)

st.markdown("### Instructions")
st.markdown("""
1. Point your index finger 🖕  
2. Raise your thumb 👍  
3. Move your hand to draw ✏️  
4. Lower your thumb to stop ❌  
""")
