import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

class FingerDrawing:
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
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255)
        }
        self.current_color = 'green'
        self.line_thickness = 3

    def is_thumb_up(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        return thumb_tip.y < thumb_ip.y
        
    def is_index_pointing(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        return index_tip.y < index_pip.y

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.drawing_app = FingerDrawing()
        self.canvas = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        if self.canvas is None:
            self.canvas = np.zeros_like(image)
        
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.drawing_app.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing_app.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.drawing_app.mp_hands.HAND_CONNECTIONS)
                
                index_finger = hand_landmarks.landmark[self.drawing_app.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = image.shape
                x, y = int(index_finger.x * w), int(index_finger.y * h)
                
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                
                if (self.drawing_app.is_thumb_up(hand_landmarks) and 
                    self.drawing_app.is_index_pointing(hand_landmarks)):
                    if self.drawing_app.last_point is None:
                        self.drawing_app.last_point = (x, y)
                    else:
                        cv2.line(self.canvas, 
                                 self.drawing_app.last_point, 
                                 (x, y), 
                                 self.drawing_app.colors[self.drawing_app.current_color], 
                                 self.drawing_app.line_thickness)
                    self.drawing_app.last_point = (x, y)
                else:
                    self.drawing_app.last_point = None

        image = cv2.addWeighted(image, 1, self.canvas, 1, 0)
        return frame.from_ndarray(image, format="bgr24")


def main():
    st.title("Finger Drawing App")
    st.write("Draw using your finger! Point your index finger and raise your thumb to draw.")
    
    # Color selection
    color = st.selectbox("Select Color", ["green", "red", "blue", "yellow"], key="color")
    
    # Setup WebRTC
    webrtc_streamer(
        key="finger-drawing",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    st.write(f"Current Color: {color}")

if __name__ == "__main__":
    main()

