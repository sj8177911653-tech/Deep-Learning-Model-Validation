
# Build a Robust Model Validation Suite with Regularization Experiments

## 1. Project Overview

This project focuses on building and evaluating a neural network model using different regularization techniques.

The main aim is to compare a baseline neural network with regularized models and understand how regularization helps reduce overfitting and improve model generalization.

The project uses the MNIST handwritten digit dataset. It contains grayscale images of handwritten digits from 0 to 9.

## 2. Problem Statement

A neural network may perform well on training data but poorly on unseen data. This problem is known as overfitting.

The objective of this project is to:
- Build a baseline neural network.
- Apply Dropout regularization.
- Apply L2 Weight Decay.
- Use Early Stopping.
- Compare model performance using learning curves.
- Evaluate all models on test data.

## 3. Dataset Description

The MNIST dataset is used in this project.

Dataset details:
- It contains 70,000 handwritten digit images.
- Each image has a size of 28 × 28 pixels.
- Images represent digits from 0 to 9.
- 60,000 images are used for training.
- 10,000 images are used for testing.

Note: MNIST was selected for this project because no specific dataset link or name was provided with the assignment.

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the MNIST dataset using TensorFlow/Keras.
2. Converted image pixel values into floating-point numbers.
3. Normalized pixel values by dividing them by 255.
4. Split the training data into training and validation sets.
5. Kept the test dataset separate for final model evaluation.

Normalization helps the neural network learn more effectively by scaling pixel values between 0 and 1.

The training data is used to train the model, validation data is used to monitor performance, and test data is used for final evaluation.

## 5. Neural Network Architecture

A neural network is a machine learning model that learns patterns from data using interconnected layers.

The architecture used in this project contains:

1. Input Layer: Accepts 28 × 28 pixel images.
2. Flatten Layer: Converts the 2D image into a 1D array.
3. First Dense Layer: Contains 128 neurons with ReLU activation.
4. Second Dense Layer: Contains 64 neurons with ReLU activation.
5. Output Layer: Contains 10 neurons with Softmax activation.

The Softmax activation function produces probabilities for the 10 digit classes.

## 6. Baseline Model

The baseline model is the basic neural network used as a reference for comparing other models.

Configuration:
- Hidden layers: 128 and 64 neurons.
- Activation function: ReLU.
- Output activation: Softmax.
- Optimizer: Adam.
- Loss function: Sparse Categorical Crossentropy.
- Evaluation metric: Accuracy.
- Epochs: 10.
- Batch size: 32.

The baseline model is trained without additional regularization techniques. Its performance is compared with the regularized models to understand the effect of regularization.

## 7. Regularization Techniques

Regularization techniques help prevent overfitting and improve the model's ability to perform well on unseen data.

### 7.1 Dropout Regularization

Dropout is a regularization technique that randomly disables some neurons during training.

Implementation:
- Dropout rate: 0.3
- Dropout layers are added after the Dense layers.
- It helps prevent the model from depending too much on particular neurons.

Benefit:
Dropout reduces overfitting and improves generalization.

### 7.2 L2 Weight Decay

L2 regularization adds a penalty to the loss function based on the squared values of model weights.

Implementation:
- L2 regularization is applied to the Dense layers.
- Regularization strength: 0.001.
- It discourages excessively large weight values.

Benefit:
L2 helps create a simpler model and reduces overfitting.

### 7.3 Early Stopping

Early Stopping is a technique that stops model training when validation performance stops improving.

Implementation:
- Monitor: Validation Loss (val_loss).
- Patience: 3 epochs.
- Restore best weights: True.
- Maximum epochs: 20.

Benefit:
Early Stopping avoids unnecessary training and helps prevent overfitting by restoring the best-performing model weights.

## 8. Model Training

All four models were trained using the MNIST training dataset.

The models used in this project are:

1. Baseline Model
2. Dropout Model
3. L2 Regularization Model
4. Early Stopping Model

Training configuration:
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy
- Batch Size: 32
- Validation Split: 20%

Each model learns patterns from training images and uses validation data to monitor its performance during training.

The trained models were saved in Keras format using the `.keras` file extension.

## 9. Learning Curves

Learning curves are graphical representations of model performance during training.

They help us understand how a model learns over different epochs.

Two types of learning curves were generated:

### 9.1 Accuracy Learning Curves

This graph displays:
- Training Accuracy
- Validation Accuracy

It helps compare how accurately each model performs on training and validation data.

Output file: `accuracy_learning_curves.png`

### 9.2 Loss Learning Curves

This graph displays:
- Training Loss
- Validation Loss

It helps identify whether the model is learning effectively and whether overfitting may be occurring.

Output file: `loss_learning_curves.png`

Both graphs compare the Baseline, Dropout, L2, and Early Stopping models.

## 10. Model Evaluation

After training, all four models were evaluated using the separate MNIST test dataset.

The purpose of evaluation is to measure how well each trained model performs on unseen data.

Evaluation metrics:
- Test Accuracy: Measures the percentage of correctly classified images.
- Test Loss: Measures the prediction error of the model.

The following models were evaluated:
1. Baseline Model
2. Dropout Model
3. L2 Regularization Model
4. Early Stopping Model

The evaluation results were generated using the `evaluation.py` script.

## 11. Results and Comparison


| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Baseline | 97.08% | 0.1331 |
| Dropout | 97.65% | 0.0841 |
| L2 Regularization | 96.79% | 0.1878 |
| Early Stopping | 97.06% | 0.0975 |

### Result Analysis

- Baseline Model achieved 97.08% test accuracy.
- Dropout Model achieved the highest test accuracy of 97.65% and the lowest test loss of 0.0841.
- L2 Model achieved 96.79% test accuracy.
- Early Stopping Model achieved 97.06% test accuracy.

Based on this evaluation run, the Dropout Model performed best among the four models.


## 12. Tools and Technologies Used

The following tools and technologies were used in this project:

- Programming Language: Python
- Deep Learning Framework: TensorFlow
- Neural Network API: Keras
- Dataset: MNIST
- Data Processing: NumPy
- Data Splitting: Scikit-learn
- Data Visualization: Matplotlib
- Development Environment: Visual Studio Code
- Model Format: Keras (.keras)

## 13. Project Files

The project contains the following important files:

| File Name | Purpose |
|---|---|
| data_preprocessing.py | Loads and preprocesses the dataset |
| baseline_model.py | Trains the baseline model |
| dropout_model.py | Trains the Dropout model |
| l2_model.py | Trains the L2 regularized model |
| early_stopping.py | Trains the model using Early Stopping |
| evaluation.py | Evaluates all trained models |
| comparison.py | Generates learning curve graphs |
| final_report.py | Generates the final evaluation text report |
| Project_Report.md | Contains the project documentation |
| README.md | Provides project information |

## 14. Conclusion

In this project, a neural network validation suite was developed using the MNIST handwritten digit dataset.

A baseline model was trained and compared with three regularization approaches: Dropout, L2 Weight Decay, and Early Stopping.

Training and validation learning curves were generated to understand model performance across epochs. All models were also evaluated using a separate test dataset.

This project demonstrates how regularization techniques can help reduce overfitting and improve the generalization ability of deep learning models.

## 15. Future Scope

- Experiment with different network architectures.
- Test different Dropout rates and L2 values.
- Use additional datasets.
- Deploy the best-performing model through a web application.
- Monitor model performance using a dashboard.
