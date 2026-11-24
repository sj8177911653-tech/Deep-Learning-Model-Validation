import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and normalize test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0

# Load all trained models
model_files = {
    "Baseline": "baseline_model.keras",
    "Dropout": "dropout_model.keras",
    "L2": "l2_model.keras",
    "Early Stopping": "early_stopping_model.keras"
}

print("\nMODEL EVALUATION RESULTS")

for name, file in model_files.items():
    model = tf.keras.models.load_model(file)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"\n{name} Model")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")