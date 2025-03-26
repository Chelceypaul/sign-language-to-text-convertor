import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset paths
dataset_path = os.path.abspath("dataset/")
processed_path = os.path.abspath("processed_data/")

# Ensure processed_data directory exists
os.makedirs(processed_path, exist_ok=True)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)

# Process each label folder
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    save_label_path = os.path.join(processed_path, label)

    os.makedirs(save_label_path, exist_ok=True)  # Ensure label-specific folder exists

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Resize and normalize image
        img = cv2.resize(img, (128, 128))
        processed_image = (img / 255.0 * 255).astype(np.uint8)  # Normalize & convert back to uint8

        # Define save path
        save_path = os.path.join(save_label_path, img_name)

        # Save the processed image
        success = cv2.imwrite(save_path, processed_image)

        if success:
            print(f"Image saved: {save_path}")
        else:
            print(f"Failed to save image: {save_path}")

print("Preprocessing Done!")
