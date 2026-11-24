import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load test dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0

models = {
    "Baseline": "baseline_model.keras",
    "Dropout": "dropout_model.keras",
    "L2 Regularization": "l2_model.keras",
    "Early Stopping": "early_stopping_model.keras"
}

with open("Final_Report.txt", "w") as report:
    report.write("DEEP LEARNING PROJECT REPORT\n")
    report.write("Robust Model Validation Suite\n\n")

    for name, file in models.items():
        model = tf.keras.models.load_model(file)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

        report.write(f"{name} Model\n")
        report.write(f"Test Loss: {loss:.4f}\n")
        report.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")

print("Final Report Generated Successfully!")