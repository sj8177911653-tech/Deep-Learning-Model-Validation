
import json
import matplotlib.pyplot as plt

histories = {
    "Baseline": "baseline_history.json",
    "Dropout": "dropout_history.json",
    "L2": "l2_history.json",
    "Early Stopping": "early_stopping_history.json"
}

# Load all model histories
model_histories = {}

for model_name, file_name in histories.items():
    with open(file_name, "r") as file:
        model_histories[model_name] = json.load(file)

# Accuracy Learning Curves
plt.figure(figsize=(12, 6))

for model_name, history in model_histories.items():
    plt.plot(history["accuracy"], label=f"{model_name} Training")
    plt.plot(history["val_accuracy"], linestyle="--",
             label=f"{model_name} Validation")

plt.title("Accuracy Learning Curves")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_learning_curves.png")
plt.show()

# Loss Learning Curves
plt.figure(figsize=(12, 6))

for model_name, history in model_histories.items():
    plt.plot(history["loss"], label=f"{model_name} Training")
    plt.plot(history["val_loss"], linestyle="--",
             label=f"{model_name} Validation")

plt.title("Loss Learning Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_learning_curves.png")
plt.show()

print("Learning Curves generated successfully!")
