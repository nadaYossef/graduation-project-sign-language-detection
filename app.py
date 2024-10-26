import cv2
import numpy as np
import logging
import time
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = "sign-language.h5"
FRAME_WIDTH = 300
FRAME_HEIGHT = 300
ROI_TOP_LEFT = (0, 40)
ROI_BOTTOM_RIGHT = (FRAME_WIDTH, FRAME_HEIGHT)
PREDICTION_THRESHOLD = 0.9
STABILITY_THRESHOLD = 5
COOLDOWN_TIME = 2
LABELS = [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
           'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank' ]

# Initialize Flask app
app = Flask(__name__)

class SignLanguageDetector:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.camera = cv2.VideoCapture(0)
        self.last_time_added = time.time()
        self.recognized_letters = []
        self.stable_prediction = None
        self.stable_count = 0
    def extract_features(self, image):
    # Ensure the image is resized to the expected model input shape (28, 28)
        feature = np.array(image).reshape(1, 28, 28, 1) / 255.0
        return feature

    def predict_letter(self, frame):
    # Convert to grayscale and resize to match the model input shape (28, 28)
        crop_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_frame_resized = cv2.resize(crop_frame_gray, (28, 28))
        feature = self.extract_features(crop_frame_resized)
        pred = self.model.predict(feature)
        return LABELS[pred.argmax()], np.max(pred)
        

    def add_letter(self, letter):
        current_time = time.time()
        if letter != self.recognized_letters[-1] if self.recognized_letters else None:
            self.recognized_letters.append(letter)
            self.last_time_added = current_time
            self.stable_count = 0
        elif current_time - self.last_time_added >= COOLDOWN_TIME:
            self.recognized_letters.append(letter)
            self.last_time_added = current_time
            self.stable_count = 0

    def process_frame(self, frame):
        # Draw the region of interest
        cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 165, 255), 1)
        crop_frame = frame[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]
        
        # Predict the letter and its confidence
        prediction_label, confidence = self.predict_letter(crop_frame)
        
        # Draw the prediction
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        if prediction_label == 'blank':
            cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            accu = "{:.2f}".format(confidence * 100)
            cv2.putText(frame, f'{prediction_label} {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Stability check and add letter if stable
        if confidence > PREDICTION_THRESHOLD:
            if self.stable_prediction == prediction_label:
                self.stable_count += 1
            else:
                self.stable_prediction = prediction_label
                self.stable_count = 1

            if self.stable_count >= STABILITY_THRESHOLD:
                self.add_letter(prediction_label)
        
        # Draw recognized letters
        recognized_text = " ".join(self.recognized_letters)
        cv2.putText(frame, recognized_text, (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame

    def gen_frames(self):
        while True:
            success, frame = self.camera.read()
            if not success:
                logging.error("Failed to read frame from camera")
                break
            frame = self.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.error("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Initialize detector
detector = SignLanguageDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detector.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
