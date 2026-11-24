import tensorflow as tf
import json
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build L2 Regularization Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu",
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(64, activation="relu",
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Save model
model.save("l2_model.keras")

print("L2 Regularization Model Training Completed!")
with open("l2_history.json", "w") as f:
    json.dump(history.history, f)
    