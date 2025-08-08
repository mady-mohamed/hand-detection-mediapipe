import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
import cv2
from collections import deque, defaultdict


HISTORY_LEN = 5
MOVEMENT_THRESHOLD = 0.01 # decrease for sensitivity, increase to ignore subtle movement
hand_x_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (0, 255, 0)
prev_landmarks = None
alpha = 0.01

cap = cv2.VideoCapture(0)
# cap is a VideoCapture object

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks # returns a list of landmarks
    handedness_list = detection_result.handedness # detects if left or right hand
    annotated_image = np.copy(rgb_image)


    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_state = "Closed" if is_hand_closed(hand_landmarks) else "Open"

        height, width, _ = annotated_image.shape
        wrist = hand_landmarks[0]
        current_x = wrist.x

        hand_label = handedness[0].category_name + f"_{idx}"
        hist = hand_x_history[hand_label]
        hist.append(current_x)
        movement = 0.0

        if len(hist) > 2:
            movement = hist[-1] - hist[-2]

        if movement > MOVEMENT_THRESHOLD:
            direction_text = "Left"
        elif movement < -MOVEMENT_THRESHOLD:
            direction_text = "Right"
        else:
            direction_text = "Still"

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList() #store landmark dat for MediaPipe
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks( # Draws landmarks on image
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name} ({hand_state}) {direction_text}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def angle_between_deg(v1, v2):
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / mag
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return np.degrees(np.arccos(cos_theta))

def is_hand_closed(hand_landmarks):
    """
    Returns True if the hand is considered closed (a fist), False otherwise.
    The logic checks if less than 3 fingers are extended.
    """
    # Landmark indices for finger joints
    pip_indices = [6, 10, 14, 18]  # Proximal Interphalangeal
    dip_indices = [7, 11, 15, 19]  # Distal Interphalangeal
    tip_indices = [8, 12, 16, 20]  # Fingertips

    extended_count = 0

    # Check the four non-thumb fingers
    for pip_idx, dip_idx, tip_idx in zip(pip_indices, dip_indices, tip_indices):
        pip = hand_landmarks[pip_idx]
        dip = hand_landmarks[dip_idx]
        tip = hand_landmarks[tip_idx]

        # Vector from PIP to DIP and DIP to TIP
        v1 = np.array([dip.x - pip.x, dip.y - pip.y, dip.z - pip.z])
        v2 = np.array([tip.x - dip.x, tip.y - dip.y, tip.z - dip.z])

        angle = angle_between_deg(v1, v2)
        # If the angle is large, the finger is considered straight (extended)
        if angle < 20:
            extended_count += 1

    # Check the thumb
    thumb_mcp = hand_landmarks[2] # Metacarpophalangeal
    thumb_ip = hand_landmarks[3]  # Interphalangeal
    index_mcp = hand_landmarks[5] # Index finger Metacarpophalangeal

    v_thumb = np.array([thumb_ip.x - thumb_mcp.x, thumb_ip.y - thumb_mcp.y, thumb_ip.z - thumb_mcp.z])
    v_to_index = np.array([index_mcp.x - thumb_mcp.x, index_mcp.y - thumb_mcp.y, index_mcp.z - thumb_mcp.z])

    thumb_angle = angle_between_deg(v_thumb, v_to_index)
    # If the angle is large, the thumb is considered extended
    if thumb_angle < 20.0:
        extended_count += 1

    # A hand is "closed" if 2 or fewer fingers are extended.
    # A hand is "open" if 3 or more fingers are extended.
    return extended_count < 3



# Load the input image directly using MediaPipe
# mp_image = mp.Image.create_from_file('woman_hands.jpg')


model_path = 'hand_landmarker.task'
image_files = ["fist.jpg", 'woman_hands.jpg', "hand_front_back.jpg"]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    # running_mode=VisionRunningMode.IMAGE
    running_mode=VisionRunningMode.VIDEO
    ,num_hands=2)

with HandLandmarker.create_from_options(options) as landmarker:
    if cap.isOpened():
        frame_timestamp = 0
        while True:
            success, frame = cap.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # hand_landmarker_result = landmarker.detect(mp_image)
            hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
            if len(hand_landmarker_result.hand_landmarks) > 0:
                curr_landmarks = hand_landmarker_result.hand_landmarks[0]
                curr_hand_landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in curr_landmarks], dtype=np.float32)
                if prev_landmarks is not None:
                    smoothed_landmarks = alpha * curr_hand_landmarks_np + (1 - alpha) * prev_landmarks
                    hand_landmarker_result.pose_landmarks[0] = smoothed_landmarks.tolist()
                    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result,
                                                              smoothed_landmarks)
                else:
                    pass


            bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hand Landmarks Cam', bgr_image)
            frame_timestamp += int(1000 / 30.0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break