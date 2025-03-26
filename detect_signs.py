import cv2
import numpy as np
import tensorflow as tf
import os
import mediapipe as mp

# Load the trained model
MODEL_PATH = "sign_language_model.h5"
assert os.path.exists(MODEL_PATH), "Error: Model file not found!"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels from the processed dataset
def get_class_labels():
    processed_path = "processed_data/"
    categories = sorted([d for d in os.listdir(processed_path) if os.path.isdir(os.path.join(processed_path, d))])
    return categories

class_labels = get_class_labels()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Get bounding box
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            w, h = x_max - x_min, y_max - y_min
            hand_label = "Right Hand" if handedness.classification[0].label == "Right" else "Left Hand"

            try:
                imgCrop = img[y_min - 20:y_max + 20, x_min - 20:x_max + 20]
                imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                imgCrop = cv2.resize(imgCrop, (128, 128))
                imgCrop = imgCrop / 255.0  # Normalize
                imgCrop = np.expand_dims(imgCrop, axis=-1)  # Ensure shape (128,128,1)
                imgCrop = np.expand_dims(imgCrop, axis=0)  # Add batch dimension (1,128,128,1)

                prediction = model.predict(imgCrop)
                predicted_label = class_labels[np.argmax(prediction)]

                # Display prediction
                cv2.rectangle(img, (x_min, y_min - 40), (x_max, y_min), (0, 255, 0), -1)
                cv2.putText(img, f"Sign: {predicted_label}", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Display hand label
                cv2.rectangle(img, (x_min, y_max), (x_max, y_max + 30), (0, 0, 255), -1)
                cv2.putText(img, hand_label, (x_min + 10, y_max + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            except Exception as e:
                print("Error processing hand image:", str(e))

    cv2.imshow("Sign Detection", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
