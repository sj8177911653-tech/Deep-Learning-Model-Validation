import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Dataset Loaded Successfully!")
print("Training Images:", x_train.shape)
print("Training Labels:", y_train.shape)
print("Testing Images:", x_test.shape)
print("Testing Labels:", y_test.shape)
# Normalize pixel values
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Data Preprocessing Completed!")
print("Training Data Range:", x_train.min(), "to", x_train.max())
print("Testing Data Range:", x_test.min(), "to", x_test.max())
from sklearn.model_selection import train_test_split

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42
)

print("Dataset Splitting Completed!")
print("Training Data:", x_train.shape)
print("Validation Data:", x_val.shape)
print("Testing Data:", x_test.shape)