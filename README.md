
# Deep Learning Project

## Project Title
Build a Robust Model Validation Suite with Regularization Experiments

## 1. Project Overview
This project builds and compares neural network models using different regularization techniques to reduce overfitting and improve generalization.

## 2. Objective
- Build a baseline neural network.
- Apply Dropout regularization.
- Apply L2 Weight Decay.
- Implement Early Stopping.
- Generate training and validation learning curves.
- Evaluate all models using test data.

## 3. Dataset
MNIST Handwritten Digit Dataset.

- Total images: 70,000
- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28 pixels
- Classes: Digits 0 to 9

Note: MNIST was selected because no specific dataset link or name was supplied with the assignment.

## 4. Models Used
1. Baseline Neural Network
2. Dropout Regularization (0.3)
3. L2 Weight Decay (0.001)
4. Early Stopping (patience=3)

## 5. Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## 6. Evaluation Metrics
- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Training Loss
- Validation Loss
- Test Loss

## 7. Model Evaluation Results

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Baseline | 97.08% | 0.1331 |
| Dropout | 97.65% | 0.0841 |
| L2 | 96.79% | 0.1878 |
| Early Stopping | 97.06% | 0.0975 |

Best-performing model: Dropout Model (97.65% accuracy).

## 8. Learning Curves
The project generates two graphs:

- `accuracy_learning_curves.png`
- `loss_learning_curves.png`

These graphs compare training and validation performance for all four models.

## 9. Installation

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## 10. How to Run

Run the scripts in this order:

1. `python data_preprocessing.py`
2. `python baseline_model.py`
3. `python dropout_model.py`
4. `python l2_model.py`
5. `python early_stopping.py`
6. `python evaluation.py`
7. `python comparison.py`
8. `python final_report.py`

## 11. Output Files
- Trained model files (`.keras`)
- Accuracy learning curves
- Loss learning curves
- `Final_Report.txt`

## 12. Conclusion
The project compares a baseline neural network with three regularization techniques. Based on the current test evaluation, the Dropout model achieved the highest test accuracy of 97.65% and the lowest test loss of 0.0841.

The project demonstrates how regularization techniques can help improve model generalization.

## 13. Future Scope
- Experiment with different neural network architectures.
- Tune regularization parameters.
- Use additional datasets.
- Deploy the best-performing model through a web application.
