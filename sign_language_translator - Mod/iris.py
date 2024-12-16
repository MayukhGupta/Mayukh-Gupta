# pip install mediapipe opencv-python pyautogui

import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False  # Use with caution
# Initialize MediaPipe Face Mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Open video capture (using webcam)
cap = cv2.VideoCapture(0)

# Constants for blink detection, cursor control, and movement amplification
BLINK_THRESHOLD = 0.25  # Adjust this threshold to fine-tune blink detection
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
smooth_factor = 5       # Higher value for smoother cursor movement
amplification_factor = 4  # Increase this value to make cursor move more than iris movement

# Helper function to calculate the eye aspect ratio (EAR) for blink detection
def calculate_eye_aspect_ratio(eye_landmarks):
    top_bottom_dist = ((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + 
                       (eye_landmarks[1].y - eye_landmarks[5].y) ** 2) ** 0.5
    left_right_dist = ((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + 
                       (eye_landmarks[0].y - eye_landmarks[3].y) ** 2) ** 0.5
    return top_bottom_dist / left_right_dist

# Previous cursor position for smoothing
prev_x, prev_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for a mirror-like view and convert color to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with Face Mesh
    results = face_mesh.process(frame_rgb)

    # Check if any face landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get iris and eye landmarks
            left_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]  # Left eye indices
            right_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]  # Right eye indices
            left_iris = face_landmarks.landmark[473]  # Left iris center
            right_iris = face_landmarks.landmark[468]  # Right iris center

            # Calculate EAR for blink detection
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            # Blink detection based on EAR threshold
            if avg_ear < BLINK_THRESHOLD:
                pyautogui.click()
            
            # Move the cursor based on iris position
            iris_x = (left_iris.x + right_iris.x) / 2
            iris_y = (left_iris.y + right_iris.y) / 2

            # Convert iris coordinates to screen position, applying amplification
            screen_x = SCREEN_WIDTH / 2 + (iris_x - 0.5) * SCREEN_WIDTH * amplification_factor
            screen_y = SCREEN_HEIGHT / 2 + (iris_y - 0.5) * SCREEN_HEIGHT * amplification_factor

            # Smooth cursor movement
            smooth_x = prev_x + (screen_x - prev_x) / smooth_factor
            smooth_y = prev_y + (screen_y - prev_y) / smooth_factor
            pyautogui.moveTo(smooth_x, smooth_y)

            # Update previous cursor position
            prev_x, prev_y = smooth_x, smooth_y

    # Display the frame
    cv2.imshow('Iris Tracking - Cursor Control', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
