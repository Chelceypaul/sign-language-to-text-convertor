import tkinter as tk
from tkinter import Label
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageTk
import mediapipe as mp

# Load trained model
MODEL_PATH = "sign_language_model.h5"
assert os.path.exists(MODEL_PATH), "Error: Model file not found!"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels dynamically from dataset
def get_class_labels():
    processed_path = "processed_data/"
    categories = sorted([d for d in os.listdir(processed_path) if os.path.isdir(os.path.join(processed_path, d))])
    return categories

class_labels = get_class_labels()

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
root = tk.Tk()
root.title("Sign Language Detection")

# Create GUI components
panel = Label(root)
panel.pack()
label_text = tk.StringVar()
label_text.set("Predicted Sign: ")
label_display = tk.Label(root, textvariable=label_text, font=("Arial", 30))
label_display.pack()

# Function to detect hands and predict sign
def detect_and_predict(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = int(min([lm.x for lm in hand_landmarks.landmark]) * img.shape[1]), int(min([lm.y for lm in hand_landmarks.landmark]) * img.shape[0])
            x_max, y_max = int(max([lm.x for lm in hand_landmarks.landmark]) * img.shape[1]), int(max([lm.y for lm in hand_landmarks.landmark]) * img.shape[0])
            
            imgCrop = img[y_min:y_max, x_min:x_max]
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgCrop = cv2.resize(imgCrop, (128, 128))
                imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                imgCrop = imgCrop / 255.0
                imgCrop = np.expand_dims(imgCrop, axis=[0, -1])
                
                prediction = model.predict(imgCrop)
                predicted_label = class_labels[np.argmax(prediction)]
                label_text.set(f"Predicted Sign: {predicted_label}")
    
# Function to update GUI frame
def update_frame():
    success, img = cap.read()
    if success:
        detect_and_predict(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
