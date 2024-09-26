# MNIST Handwritten Digit Classification

This project focuses on classifying handwritten digits from the popular **MNIST dataset** using two machine learning models:
1. **Artificial Neural Network (ANN)**
2. **Decision Tree Classifier**

## Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels in size. It is widely used for training and testing image processing systems. In this project, I have implemented and compared the performance of an **ANN** model and a **Decision Tree** classifier to recognize these digits.

## Models

### 1. Artificial Neural Network (ANN)

The ANN model is built using a simple feed-forward neural network architecture, which includes:
- **Input Layer**: 784 neurons (28x28 pixel images flattened).
- **Hidden Layers**: Fully connected layers with ReLU activation.
- **Output Layer**: 10 neurons (one for each digit) using softmax activation.

The model is trained using **cross-entropy loss** and optimized with **Adam** optimizer to achieve high accuracy on the test set.

### 2. Decision Tree Classifier

The Decision Tree model is used as a baseline classifier. It is a non-parametric supervised learning algorithm that splits the dataset based on the most significant features to classify the digits. 

Though simpler compared to the ANN, it provides an interpretable comparison of how traditional algorithms perform on image data.

## Dataset

The **MNIST** dataset is available through many sources, including `Keras.datasets`. It includes:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

Each image is labeled with a digit from 0 to 9.

## Implementation

The project is implemented using **Python** with the following libraries:
- **TensorFlow/Keras**: For building and training the ANN model.
- **Scikit-learn**: For building the Decision Tree model.
- **Matplotlib & Seaborn**: For data visualization.
- **Numpy & Pandas**: For data manipulation.

