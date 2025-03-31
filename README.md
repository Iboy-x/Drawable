# Finger Drawing Application

A real-time drawing application that allows users to draw using hand gestures. The application uses computer vision to track hand movements and enables drawing through finger tracking.

## Features

- Real-time hand tracking
- Drawing with index finger and thumb gestures
- Multiple color options
- Clear canvas functionality
- Download drawings as PNG files
- Web-based interface

## Requirements

- Python 3.7 or higher
- Webcam
- Modern web browser (for web version)

## Dependencies

- `opencv-python`: For camera handling and image processing
- `mediapipe`: For hand tracking and gesture recognition
- `numpy`: For numerical operations and array handling
- `streamlit`: For web interface (web version only)
- `Pillow`: For image processing (web version only)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
python -m pip install -r requirements.txt
```

## Usage

### Desktop Version

Run the desktop version:
```bash
python finger_drawing.py
```

Controls:
- Point index finger and raise thumb to draw
- Lower thumb to stop drawing
- Press 'c' to clear canvas
- Press '1-4' to change colors:
  - 1: Green
  - 2: Red
  - 3: Blue
  - 4: Yellow
- Press 'q' to quit

### Web Version

Run the web version:
```bash
python -m streamlit run app.py
```

Features:
- Color selection dropdown
- Clear canvas button
- Download drawing button
- Real-time camera feed
- Hand tracking visualization

## Project Structure

```
.
├── app.py              # Web version of the application
├── finger_drawing.py   # Desktop version of the application
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Technical Details

### Hand Tracking
The application uses MediaPipe's hand tracking to detect:
- Index finger position for drawing
- Thumb position for drawing mode activation
- Hand landmarks for visualization

### Drawing Process
1. Camera captures video feed
2. Each frame is processed for hand detection
3. When thumb is up and index finger is pointing:
   - Drawing mode is activated
   - Line is drawn from previous point to current point
4. Drawing continues until gesture is changed

### Color System
- Colors are defined in RGB format
- Default colors: green, red, blue, yellow
- Line thickness is configurable

## Browser Support

The web version is compatible with:
- Chrome (recommended)
- Firefox
- Edge
- Safari

## Notes

- Ensure good lighting for better hand detection
- Keep hand within camera frame
- Allow camera access when prompted
- Web version requires stable internet connection

## License

This project is open source and available under the MIT License. 