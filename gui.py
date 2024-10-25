import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox
import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from PIL import Image, ImageTk

# Load the trained model
model = tf.keras.models.load_model("sign-language-mnist.h5")

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Define function to preprocess the input image
def preprocess_image(image):
    img_resized = cv2.resize(image, (28, 28))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.reshape(1, 28, 28, 1).astype('float32') / 255.0
    return img_gray

# Function to predict the letter from the image
def predict_image(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    predicted_label = np.argmax(prediction)
    return chr(65 + predicted_label)  # Convert to corresponding ASCII letter

# Define function for text-to-speech
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Define function for image upload and prediction
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        letter = predict_image(image)
        result_label.config(text=f"Predicted Letter: {letter}")
        speak_text(letter)
        history_text.insert(tk.END, f"{letter} ")

# Function to start the video capture and real-time prediction
def start_video_capture():
    cap = cv2.VideoCapture(0)
    result_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture video.")
            break

        frame_flipped = cv2.flip(frame, 1)
        roi = frame_flipped[100:400, 100:400]
        letter = predict_image(roi)

        # Display results on video frame
        cv2.rectangle(frame_flipped, (100, 100), (400, 400), (255, 0, 0), 2)
        cv2.putText(frame_flipped, f"Predicted Letter: {letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the video frame
        cv2.imshow("Real-Time Sign Language Prediction", frame_flipped)

        # Append predicted letter to the result text
        result_text += letter
        history_text.insert(tk.END, f"{letter} ")

        # Text-to-speech for every new letter
        speak_text(letter)

        # Press 'q' to exit the video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    result_label.config(text=f"Predicted Text: {result_text}")

# Main application window
app = tk.Tk()
app.title("Sign Language to Text and Speech Converter")
app.geometry("600x400")

# Frame for instructions
instruction_frame = Frame(app, pady=10)
instruction_frame.pack()
instructions = Label(instruction_frame, text="Instructions:\n1. Upload an image to predict a letter.\n"
                                             "2. Start video for real-time letter prediction.\n"
                                             "3. 'q' key stops the video capture.", font=("Arial", 10))
instructions.pack()

# Buttons for uploading image and starting video capture
upload_button = Button(app, text="Upload Image", command=upload_image, width=15)
upload_button.pack(pady=10)
video_button = Button(app, text="Start Video Capture", command=start_video_capture, width=15)
video_button.pack(pady=10)

# Result label to display predicted letters
result_label = Label(app, text="Predicted Text: ", font=("Arial", 14))
result_label.pack(pady=10)

# History text field to show the sequence of predictions
history_frame = Frame(app)
history_frame.pack()
history_label = Label(history_frame, text="Prediction History:", font=("Arial", 12))
history_label.pack()
history_text = tk.Text(history_frame, height=5, width=50)
history_text.pack()

# Run the app
app.mainloop()
