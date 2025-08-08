# Hand Tracking with MediaPipe

This project uses [MediaPipe](https://developers.google.com/mediapipe) Hand Landmarker to:
- Detect and draw 21 3D hand landmarks on live video.
- Classify hand state as **Open** or **Closed** (fist) based on finger joint angles.
- Detect simple horizontal movement (`Left`, `Right`, `Still`) of each hand.
- Display handedness (Left / Right) in real time.

## Features
- **Real-time detection** from webcam feed using MediaPipe's `hand_landmarker.task`.
- **Finger extension logic**:
  - Calculates angles between finger joints to determine whether each finger is extended.
  - Special handling for thumb angle.
- **Open/Closed classification**:
  - Open = ≥ 3 extended fingers.
  - Closed = < 3 extended fingers.
- **Movement direction detection** using wrist horizontal position history.
  - Tracks wrist X-position over the last few frames (`deque` with `HISTORY_LEN`).
  - **Exponential smoothing** is applied to wrist position data to reduce jitter and make movement direction detection more stable.
    - Formula: `smoothed = α * current + (1 - α) * previous`
    - Lower `α` → smoother but more delayed response.
  - Compares latest two smoothed positions:
    - `Right` if movement > threshold
    - `Left` if movement < -threshold
    - `Still` otherwise

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- MediaPipe with Tasks API (Vision module)

## Installation

1. **Clone or download this repository**  
   ```bash
   git clone https://github.com/your-username/hand-tracking-mediapipe.git
   cd hand-tracking-mediapipe
   ```

2. **Install dependencies**  
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. **Download the MediaPipe Hand Landmarker model**  
   - Get `hand_landmarker.task` from the official MediaPipe models repository:  
     https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
   - Place it in the project root directory.

## Usage

Run the script:
```bash
python hand_tracking.py
```

Controls:
- Press **Q** to quit the webcam window.

## How It Works

### Landmark Detection
- MediaPipe Hand Landmarker returns 21 normalized coordinates for each detected hand.
- These points represent knuckles and tips of all fingers, including the thumb.

### Finger Extension Detection
- **Non-thumb fingers**:  
  - Compares angle between PIP→DIP and DIP→TIP segments.  
  - If angle is small (finger is straight), counts as extended.
- **Thumb**:  
  - Compares MCP→IP with MCP→Index MCP vector.  
  - If angle is small, counts as extended.

### Hand Open/Closed Logic
- Count extended fingers:
  - **Open** if `extended_count >= 3`
  - **Closed** if `extended_count < 3`

### Movement Detection
- Tracks wrist X-position over the last few frames (`deque` with `HISTORY_LEN`).
- Compares latest two positions:
  - `Right` if movement > threshold
  - `Left` if movement < -threshold
  - `Still` otherwise

## File Structure
```
hand_tracking.py
hand_landmarker.task
README.md
```

## Example Output
The webcam feed will display:
- Landmarks and hand skeleton.
- Handedness (Left/Right), state (Open/Closed), and movement direction.

Example:
```
Right (Open) Left
```

---

## Notes
- Adjust `MOVEMENT_THRESHOLD` and angle thresholds in `is_hand_closed()` for your needs.
- Works with 1 or 2 hands in frame.
- Tested with `mediapipe` v0.10+.

## License
This project is under the MIT License.
```
