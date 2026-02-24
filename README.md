# Animal Image Classification Using Neural Networks

## Project Overview
This project presents a deep learning pipeline for multi-class animal image classification using Convolutional Neural Networks (CNNs).
The objective is to classify images into one of five categories:
 - Cat
 - Dog
 - Horse
 - Elephant
 - Lion
The model was implemented in Python using TensorFlow/Keras and trained from scratch.


## Problem Statement
Image classification is a core problem in computer vision.
The goal of this project is to build a robust CNN model capable of learning discriminative visual features and accurately classifying animal images, even in the presence of:
  - Class imbalance
  - Intra-class variability
  - Visually similar categories (e.g., Cat vs. Lion)
The project demonstrates a complete deep learning workflow, from data preprocessing to final evaluation.


## Dataset
Used dataset: https://www.kaggle.com/datasets/shiv28/animal-5-mammal

The dataset is organized into:
train/ (used for training and validation split)
test/ (used strictly for final evaluation)

- Preprocessing steps:
   - Image resizing to 64 × 64
   - Batch size: 64
   - Automatic training/validation split (80/20)
   - Pixel normalization using Rescaling(1./255)
   - Removal of corrupted images (JFIF validation script)
   - Visualization of dataset samples


## Handling Class Imbalance
Since the dataset classes are imbalanced, class weights were computed using:
sklearn.utils.class_weight
These weights were passed to the training process to ensure fair contribution of minority classes and prevent bias toward dominant categories.
Class distribution was analyzed and visualized prior to training.


## Data Augmentation
To improve generalization and reduce overfitting, data augmentation was applied:
  - Random horizontal flip
  - Random rotation (±25%)
  - Random zoom
This augmentation was implemented directly inside the model pipeline using Keras preprocessing layers.


## Model Architecture
The model is a custom Convolutional Neural Network built using the Keras Sequential API.

Architecture:
Data Augmentation Layer
Rescaling Layer
Conv2D (16 filters) + ReLU
MaxPooling
Conv2D (32 filters) + ReLU
MaxPooling
Conv2D (64 filters) + ReLU
MaxPooling
Dropout (0.2)
Flatten
Dense (128 units) + Activation
L2 Regularization
Softmax Output Layer (5 classes)

Training Setup:
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metric: Accuracy


## Hyperparameter Tuning
The following hyperparameters were experimentally analyzed:
  - Learning rate
  - Activation functions
  - L2 regularization strength
  - Number of training epochs
Model performance was monitored using:
  - Training and validation accuracy
  - Training and validation loss
  - Confusion matrices
  - Validation accuracy score


## Model Evaluation
Evaluation was performed on:
 - Validation set (during training)
 Independent test set (final evaluation)

Metrics used:
  - Accuracy score
  - Normalized confusion matrix
  - Visualization of misclassified images
The confusion matrix analysis highlights the model’s performance across all five animal classes and reveals where most classification errors occur.


## Technologies Used
  - Python
  - TensorFlow
  - Keras
  - NumPy
  - Matplotlib
  - Scikit-learn
