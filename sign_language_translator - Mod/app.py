from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import numpy as np
import mediapipe as mp
import pickle
import warnings
from collections import deque, Counter
import os

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

app = Flask(__name__)

# Load model
model_path = r'C:\Users\hp\mayukh\other\Project\sign_language_translator\model.p'  # Ensure this path is correct
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8)

current_word = []
predictionletter = ""
recent_predictions = deque(maxlen=5)  # Store the last 5 predictions

CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for accepting predictions

def get_majority_prediction(predictions):
    if not predictions:
        return ""
    count = Counter(predictions)
    return count.most_common(1)[0][0]

def gen_frames():
    global predictionletter  # Ensure we are modifying the global variable
    cap = cv2.VideoCapture(1)# change to 1 to use webcam 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower the resolution if needed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    last_prediction = ""
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)
            results = hands.process(frame_rgb)
            prediction = ""
            prediction_prob = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(236, 255, 3), thickness=2, circle_radius=2)
                    )

                    data_aux = []
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x)
                        data_aux.append(landmark.y)

                    if data_aux:
                        prediction_prob = max(model.predict_proba([np.array(data_aux)[:42]])[0])
                        prediction = model.predict([np.array(data_aux)[:42]])[0]
                        predictionletter = prediction  # Update the global variable
                        #print(predictionletter)  # Debugging

            if prediction and prediction_prob > CONFIDENCE_THRESHOLD:
                recent_predictions.append(prediction)
                majority_prediction = get_majority_prediction(recent_predictions)

                if majority_prediction and majority_prediction != last_prediction:
                    current_word.append(majority_prediction)
                    last_prediction = majority_prediction
                    if len(current_word) > 30:  # Limit the length of the word
                        current_word.pop(0)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_word')
def get_current_word():
    return jsonify({'word': ''.join(current_word)})

@app.route('/prediction')
def get_prediction():
    global predictionletter  # Ensure we are accessing the global variable
#    print(predictionletter)  # Debugging
    return jsonify({'prediction': predictionletter})

@app.route('/reset_word')
def reset_word():
    global current_word
    current_word = []
    return jsonify({'status': 'reset'})

@app.route('/image.png')
def serve_image():
    return send_from_directory('/image', 'image.png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
