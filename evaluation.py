import tensorflow as tf
import numpy as np
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load trained model
MODEL_PATH = "sign_language_model.h5"
assert os.path.exists(MODEL_PATH), "Error: Model file not found!"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels dynamically
PROCESSED_DATA_PATH = "processed_data/"
assert os.path.exists(PROCESSED_DATA_PATH), "Error: Processed data directory not found!"

class_labels = sorted([d for d in os.listdir(PROCESSED_DATA_PATH) if os.path.isdir(os.path.join(PROCESSED_DATA_PATH, d))])
if not class_labels:
    raise ValueError("Error: No class labels found! Ensure 'processed_data/' contains labeled subdirectories.")

print("Loaded Class Labels:", class_labels)
print("Number of Classes:", len(class_labels))

# Function to preprocess image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=[0, -1])  # Reshape for model
    return img

# Collect test images
test_images = []
true_labels = []
for label in class_labels:
    label_path = os.path.join(PROCESSED_DATA_PATH, label)
    for img_file in os.listdir(label_path):
        if img_file.endswith(('.jpg', '.png')):
            test_images.append(os.path.join(label_path, img_file))
            true_labels.append(label)

# Ensure there are test images
if not test_images:
    raise ValueError("Error: No test images found! Ensure 'processed_data/' contains images.")

# Predict test images
predicted_labels = []
for img_path in test_images:
    img = preprocess_image(img_path)
    if img is None:
        continue
    prediction = model.predict(img)

    # Debugging: Print the shape of the prediction output
    print(f"Prediction shape for {img_path}: {prediction.shape}")

    # Ensure the prediction output matches the number of classes
    if prediction.shape[1] != len(class_labels):
        raise ValueError(f"Mismatch! Model predicts {prediction.shape[1]} classes, but found {len(class_labels)} labels.")

    predicted_label = class_labels[np.argmax(prediction)]
    predicted_labels.append(predicted_label)

# Convert labels to numerical values for evaluation
true_labels_numeric = [class_labels.index(label) for label in true_labels]
predicted_labels_numeric = [class_labels.index(label) for label in predicted_labels]

# Accuracy, Precision, Recall, F1-score
print("Classification Report:")
print(classification_report(true_labels_numeric, predicted_labels_numeric, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_labels_numeric, predicted_labels_numeric)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
