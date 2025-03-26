import pytest
import tensorflow as tf
import numpy as np
import cv2
import os
import random

tf.get_logger().setLevel('ERROR')

MODEL_PATH = "sign_language_model.h5"
PROCESSED_DATA_PATH = "processed_data/"

# Load the trained model
@pytest.fixture(scope="session")
def load_model():
    if not os.path.exists(MODEL_PATH):
        pytest.fail("Error: Model file 'sign_language_model.h5' not found!")
    return tf.keras.models.load_model(MODEL_PATH)

# Extract class labels from dataset directory names
def get_class_labels():
    if not os.path.exists(PROCESSED_DATA_PATH):
        pytest.fail("Error: 'processed_data/' directory is missing!")
    
    categories = [d for d in os.listdir(PROCESSED_DATA_PATH) if os.path.isdir(os.path.join(PROCESSED_DATA_PATH, d))]
    
    if not categories:
        pytest.fail("Error: No categories found in 'processed_data/'!")
    
    return categories  # Class names are folder names

# Dynamically select random images from each category
def get_test_images():
    if not os.path.exists(PROCESSED_DATA_PATH):
        return []
    
    test_images = []
    categories = get_class_labels()
    
    for category in categories:
        category_path = os.path.join(PROCESSED_DATA_PATH, category)
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png'))]
        if images:
            test_images.append(os.path.join(category_path, random.choice(images)))

    return test_images if len(test_images) >= 3 else []

# Preprocess test images before passing to the model
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        pytest.skip(f"Skipping test: Test image '{image_path}' not found!")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pytest.fail(f"Error: Failed to load image '{image_path}'!")

    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=[0, -1])  # Reshape for model input
    return img

# Test if the model loads correctly
def test_model_loading(load_model):
    assert load_model is not None, "Error: Model failed to load!"

# Test image preprocessing pipeline
@pytest.mark.parametrize("image_path", get_test_images())
def test_image_preprocessing(image_path):
    processed_img = preprocess_image(image_path)
    assert processed_img.shape == (1, 128, 128, 1), "Error: Preprocessed image shape is incorrect!"
    assert np.all((processed_img >= 0.0) & (processed_img <= 1.0)), "Error: Image not normalized correctly!"

# Test model prediction and output validity
@pytest.mark.parametrize("image_path", get_test_images())
def test_model_prediction(load_model, image_path):
    class_labels = get_class_labels()
    if len(class_labels) == 0:
        pytest.skip("Skipping test: No class labels found!")

    processed_img = preprocess_image(image_path)
    prediction = load_model.predict(processed_img)

    assert prediction.shape == (1, len(class_labels)), "Error: Prediction output does not match class count!"
    assert np.allclose(np.sum(prediction), 1.0, atol=1e-3), "Error: Softmax output does not sum to 1!"

    predicted_label = class_labels[np.argmax(prediction)]
    print(f"Predicted Label for {image_path}: {predicted_label}")
