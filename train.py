import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Parameters
img_size = 128
batch_size = 32
epochs = 20  # Increased for better training
processed_path = os.path.abspath("processed_data/")

# Ensure processed_data exists
if not os.path.exists(processed_path):
    raise FileNotFoundError("Error: 'processed_data/' directory is missing! Run preprocessing first.")

# Data Augmentation
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

# Load datasets
train_data = datagen.flow_from_directory(processed_path, target_size=(img_size, img_size),
                                         color_mode="grayscale", batch_size=batch_size,
                                         class_mode="categorical", subset="training")

val_data = datagen.flow_from_directory(processed_path, target_size=(img_size, img_size),
                                       color_mode="grayscale", batch_size=batch_size,
                                       class_mode="categorical", subset="validation")

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # Extra Conv layer for better feature extraction
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks (to improve training)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
]

# Train Model
model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)

# Save Final Model
model.save("sign_language_model.h5")
print("Model Trained and Saved Successfully!")
