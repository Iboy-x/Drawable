import cv2
import mediapipe as mp
import numpy as np
import time
import os

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
        
    def list_cameras(self):
        """List all available cameras"""
        print("\nChecking available cameras...")
        available_cameras = []
        
        # Try to find cameras in the system
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Found camera at index {i}")
                    available_cameras.append(i)
                cap.release()
        
        return available_cameras
        
    def get_smooth_point(self, point):
        """Apply smoothing to the point coordinates"""
        self.smooth_points.append(point)
        if len(self.smooth_points) > self.smooth_length:
            self.smooth_points.pop(0)
        
        if len(self.smooth_points) < 2:
            return point
            
        # Calculate average point
        avg_x = sum(p[0] for p in self.smooth_points) / len(self.smooth_points)
        avg_y = sum(p[1] for p in self.smooth_points) / len(self.smooth_points)
        return (int(avg_x), int(avg_y))
        
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
        
    def start(self):
        # Initialize camera
        print("Initializing camera...")
        
        # First, list all available cameras
        available_cameras = self.list_cameras()
        
        if not available_cameras:
            print("\nError: No cameras found!")
            print("\nTroubleshooting steps:")
            print("1. Open Windows Settings")
            print("2. Go to Privacy & Security > Camera")
            print("3. Make sure 'Camera access' is turned ON")
            print("4. Make sure 'Let apps access your camera' is turned ON")
            print("5. Check if your camera is enabled in Device Manager")
            print("6. Try running the program as administrator")
            return
            
        # Try to use the first available camera
        print(f"\nUsing camera at index {available_cameras[0]}")
        cap = cv2.VideoCapture(available_cameras[0])
        
        if not cap.isOpened():
            print("Error: Could not open the selected camera!")
            return
            
        print("Camera initialized successfully!")
        print("\nControls:")
        print("- Point index finger and raise thumb to draw")
        print("- Lower thumb to stop drawing")
        print("- Press 'c' to clear canvas")
        print("- Press '1-4' to change colors (1:green, 2:red, 3:blue, 4:yellow)")
        print("- Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera!")
                break
                
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(rgb_frame)
            
            # Initialize canvas if not done yet
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)
            
            # Draw the hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get index finger tip coordinates
                    index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, c = frame.shape
                    x, y = int(index_finger.x * w), int(index_finger.y * h)
                    
                    # Check if thumb is up and index is pointing
                    if self.is_thumb_up(hand_landmarks) and self.is_index_pointing(hand_landmarks):
                        self.drawing_mode = True
                        if self.last_point is None:
                            self.last_point = (x, y)
                        else:
                            cv2.line(self.canvas, self.last_point, (x, y), 
                                   self.colors[self.current_color], self.line_thickness)
                        self.last_point = (x, y)
                    else:
                        self.drawing_mode = False
                        self.last_point = None
            
            # Draw status indicators
            status_y = 30
            # Draw color indicator
            cv2.putText(frame, f"Color: {self.current_color}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[self.current_color], 2)
            # Draw drawing mode indicator
            status_y += 30
            cv2.putText(frame, f"Drawing: {'ON' if self.drawing_mode else 'OFF'}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine the frame with the canvas
            frame = cv2.addWeighted(frame, 1, self.canvas, 1, 0)
            
            # Display the frame
            cv2.imshow('Finger Drawing', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)
            elif key == ord('1'):
                self.current_color = 'green'
            elif key == ord('2'):
                self.current_color = 'red'
            elif key == ord('3'):
                self.current_color = 'blue'
            elif key == ord('4'):
                self.current_color = 'yellow'
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Check if running with administrator privileges
        if os.name == 'nt':  # Windows
            try:
                is_admin = os.getuid() == 0
            except AttributeError:
                import ctypes
                is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
                
            if not is_admin:
                print("Note: This program might need administrator privileges to access the camera.")
                print("Try running the program as administrator if you continue to have issues.")
        
        app = FingerDrawing()
        app.start()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure you have all the required dependencies installed correctly.") 