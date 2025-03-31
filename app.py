import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

class FingerDrawing:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize drawing variables
        self.drawing_mode = False
        self.last_point = None
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255)
        }
        self.current_color = 'green'
        self.line_thickness = 3
        
        # Initialize canvas
        self.canvas = None
        
    def is_thumb_up(self, hand_landmarks):
        """Simplified thumb up detection"""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        return thumb_tip.y < thumb_ip.y
        
    def is_index_pointing(self, hand_landmarks):
        """Check if index finger is pointing"""
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        return index_tip.y < index_pip.y

def main():
    st.title("Finger Drawing App")
    st.write("Draw using your finger! Point your index finger and raise your thumb to draw.")
    
    # Initialize session state
    if 'canvas' not in st.session_state:
        st.session_state.canvas = None
    if 'drawing_app' not in st.session_state:
        st.session_state.drawing_app = FingerDrawing()
    
    # Color selection
    color = st.selectbox(
        "Select Color",
        ["green", "red", "blue", "yellow"],
        key="color"
    )
    st.session_state.drawing_app.current_color = color
    
    # Clear button
    if st.button("Clear Canvas"):
        st.session_state.canvas = None
    
    # Camera input
    camera_input = st.camera_input("Take a picture")
    
    if camera_input is not None:
        # Convert the image to numpy array
        image = Image.open(camera_input)
        frame = np.array(image)
        
        # Initialize canvas if needed
        if st.session_state.canvas is None:
            st.session_state.canvas = np.zeros_like(frame)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = st.session_state.drawing_app.hands.process(rgb_frame)
        
        # Draw the hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                st.session_state.drawing_app.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks,
                    st.session_state.drawing_app.mp_hands.HAND_CONNECTIONS
                )
                
                # Get index finger tip coordinates
                index_finger = hand_landmarks.landmark[st.session_state.drawing_app.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = frame.shape
                x, y = int(index_finger.x * w), int(index_finger.y * h)
                
                # Draw a circle at the index finger tip for better visibility
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                
                # Check if thumb is up and index is pointing
                if (st.session_state.drawing_app.is_thumb_up(hand_landmarks) and 
                    st.session_state.drawing_app.is_index_pointing(hand_landmarks)):
                    if st.session_state.drawing_app.last_point is None:
                        st.session_state.drawing_app.last_point = (x, y)
                    else:
                        # Draw line on canvas
                        cv2.line(st.session_state.canvas, 
                                st.session_state.drawing_app.last_point, 
                                (x, y), 
                                st.session_state.drawing_app.colors[color], 
                                st.session_state.drawing_app.line_thickness)
                    st.session_state.drawing_app.last_point = (x, y)
                else:
                    st.session_state.drawing_app.last_point = None
        
        # Combine the frame with the canvas
        frame = cv2.addWeighted(frame, 1, st.session_state.canvas, 1, 0)
        
        # Convert back to PIL Image for display
        image = Image.fromarray(frame)
        st.image(image, caption="Drawing Canvas", use_column_width=True)
        
        # Add download button
        if st.button("Download Drawing"):
            # Convert canvas to PIL Image
            canvas_image = Image.fromarray(st.session_state.canvas)
            # Save to bytes
            img_byte_arr = io.BytesIO()
            canvas_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            # Create download button
            st.download_button(
                label="Download Drawing",
                data=img_byte_arr,
                file_name="drawing.png",
                mime="image/png"
            )
        
        # Add status indicators
        st.write("Status:")
        st.write(f"Drawing Mode: {'ON' if st.session_state.drawing_app.last_point is not None else 'OFF'}")
        st.write(f"Current Color: {color}")
        st.write("Instructions:")
        st.write("1. Point your index finger")
        st.write("2. Raise your thumb")
        st.write("3. Move your index finger to draw")
        st.write("4. Lower your thumb to stop drawing")

if __name__ == "__main__":
    main() 
