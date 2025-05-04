import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import io

class FingerDrawing(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_point = None
        self.canvas = None
        self.color = (0, 255, 0)
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

                index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger.x * w), int(index_finger.y * h)

                if self.is_thumb_up(hand_landmarks) and self.is_index_pointing(hand_landmarks):
                    if self.last_point is None:
                        self.last_point = (x, y)
                    else:
                        cv2.line(self.canvas, self.last_point, (x, y), self.color, self.line_thickness)
                        self.last_point = (x, y)
                else:
                    self.last_point = None

        img = cv2.addWeighted(img, 1, self.canvas, 1, 0)
        return img

# --- Streamlit UI ---
st.title("🖐️ Finger Drawing in Real-Time")
st.write("Raise your thumb and point your index finger to draw in the air!")

# Color selection
color_name = st.selectbox("🎨 Pick a color", ["Green", "Red", "Blue", "Yellow"])
color_dict = {
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255)
}

# Init transformer and pass the selected color
transformer = FingerDrawing()
transformer.color = color_dict[color_name]

webrtc_ctx = webrtc_streamer(key="draw", video_transformer_factory=lambda: transformer)

# Clear canvas
if st.button("🧹 Clear Canvas"):
    transformer.canvas = None
    transformer.last_point = None

# Download button
if transformer.canvas is not None:
    canvas_pil = Image.fromarray(transformer.canvas)
    img_byte_arr = io.BytesIO()
    canvas_pil.save(img_byte_arr, format='PNG')
    st.download_button("📥 Download Drawing", data=img_byte_arr.getvalue(), file_name="drawing.png", mime="image/png")

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    1. Allow camera access when prompted.
    2. Raise your thumb and point your index finger.
    3. Move your hand to draw.
    4. Lower thumb or unpoint finger to stop drawing.
    """)

