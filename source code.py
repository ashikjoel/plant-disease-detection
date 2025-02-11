from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os

zip_file_path = "/content/drive/MyDrive/archive.zip"
extract_dir = "/content"

if os.path.exists(zip_file_path):
    try:
        # Create a ZipFile object
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all contents to the specified directory
            zip_ref.extractall(extract_dir)
        print(f"Successfully extracted '{zip_file_path}' to '{extract_dir}'")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' is not a valid zip file.")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print(f"Error: '{zip_file_path}' does not exist.")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pathlib

import os

import glob as gb

import glob

import cv2

import tensorflow as tfimport tensorflow as tf

# Define the size for resizing images
size = 224  # Replace 224 with your desired dimension

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1/255.0,
    preprocessing_function=None,
    validation_split=0.1  # Reserving 10% of data for validation
).flow_from_directory(
    directory="/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train",  # Replace with the actual path to your training dataset
    batch_size=164,
    target_size=(size, size),  # Resize images to (224, 224) or as defined
    subset="training",
    color_mode='rgb',  # Can be "rgb", "rgba", or "grayscale"
    class_mode='categorical',  # Use 'binary', 'sparse', 'categorical' or None as needed
    shuffle=True
)import tensorflow as tf  # Import TensorFlow

valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    preprocessing_function=None,
    validation_split=0.1
).flow_from_directory(
    directory="/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid",
    batch_size=164,
    target_size=(224, 224),
    subset='validation',
    color_mode='rgb',  # "rgb", "rgba", or "grayscale"
    class_mode='categorical',  # Use 'binary', 'sparse', 'categorical' or None as needed
    shuffle=False
)
# Path to the dataset
test = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid'

# Test generator
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    preprocessing_function=None
).flow_from_directory(
    directory=test,
    batch_size=164,
    target_size=(224, 224),
    color_mode='rgb',  # Choose "rgb", "rgba", or "grayscale"
    class_mode='categorical',  # Use 'binary', 'sparse', 'categorical', or None as needed
    shuffle=False
)

from tensorflow import keras

# Initialize the Sequential model
model = keras.models.Sequential()  # To build the Neural Network

# Add layers to the model
model.add(keras.layers.Conv2D(
    filters=32, kernel_size=7, strides=1, padding="same",
    activation="relu", name="Conv1", input_shape=(224, 224, 3)
))
model.add(keras.layers.MaxPool2D(pool_size=2, name="Pool1"))

model.add(keras.layers.Conv2D(
    filters=64, kernel_size=5, strides=1, padding="same",
    activation="relu", name="Conv2"
))
model.add(keras.layers.MaxPool2D(pool_size=2, name="Pool2"))

model.add(keras.layers.Conv2D(
    filters=128, kernel_size=3, strides=1, padding="same",
    activation="relu", name="Conv3"
))
model.add(keras.layers.Conv2D(
    filters=256, kernel_size=3, strides=1, padding="same",
    activation="relu", name="Conv4"
))
model.add(keras.layers.MaxPool2D(pool_size=2, name="Pool3"))

# Flatten layer to convert into a 1D vector
model.add(keras.layers.Flatten(name="Flatten1"))

# Fully connected layers
model.add(keras.layers.Dense(128, activation="relu", name="Dense1"))
model.add(keras.layers.Dropout(0.5))  # Dropout layer

model.add(keras.layers.Dense(64, activation="relu", name="Dense2"))
model.add(keras.layers.Dropout(0.5))  # Dropout layer

# Output layer (38 classes with softmax activation)
model.add(keras.layers.Dense(38, activation="softmax", name="Output"))

# Display the model's summary
print(model.summary())

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Define ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True
)

# Define ReduceLROnPlateau callback
model_reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=15,
    min_lr=0.000001
)

# List of callbacks
callbacks = [early_stopping, model_checkpoint, model_reduce_lr_on_plateau]
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
)


# Train the model
history = model.fit(
    train_generator,
    epochs=5,  # Set the desired number of epochs
    validation_data=valid_generator,
    callbacks=callbacks
)

# Evaluate the model
model_evaluate = model.evaluate(test_generator)

# Print evaluation metrics
print("Loss:", model_evaluate[0])
print("Accuracy:", model_evaluate[1])
print("Precision:", model_evaluate[2])
print("Recall:", model_evaluate[3])

model.save('CNN_plantdiseases_model.keras')