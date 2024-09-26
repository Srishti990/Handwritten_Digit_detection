# MNIST Handwritten Digit Classification

This project focuses on classifying handwritten digits from the famous MNIST dataset using two different models: an Artificial Neural Network (ANN) and a Random Forest classifier. The aim is to compare the performance of these models in recognizing handwritten digits.

## Project Overview

The MNIST dataset consists of 60,000 images of handwritten digits (0–9). Each image is 28x28 pixels, and the goal is to correctly classify each image as the corresponding digit.

### Models Implemented

1. **Artificial Neural Network (ANN)**:
   - A fully connected deep neural network was implemented using TensorFlow/Keras. The network learns features directly from the pixel data and classifies the digit images.

2. **Random Forest**:
   - A machine learning algorithm known for its effectiveness on structured data. The Random Forest model uses decision trees on random subsets of the data to predict the correct digit.

## Dataset

The MNIST dataset consists of grayscale images of size 28x28 pixels with labels representing the digits 0–9.

| Data Split   | Number of Samples |
|--------------|-------------------|
| Training Set | 48,000            |
| Test Set     | 12,000            |

Each image has 784 features (28x28 pixels flattened into a single array), and the labels are integers from 0 to 9 corresponding to the digit shown in the image.

## Libraries Used

- **TensorFlow/Keras**: For building and training the ANN model.
- **Scikit-learn**: For the Random Forest implementation.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Seaborn**: For enhanced visualizations.
- **NumPy**: For numerical computations.

## Model Architectures

### 1. Artificial Neural Network (ANN)
- **Input Layer**: 784 neurons (28x28 pixels)
- **Hidden Layers**:
  - Layer 1: 512 neurons, ReLU activation
  - Layer 2: 256 neurons, ReLU activation
- **Output Layer**: 10 neurons (one for each digit), softmax activation
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Evaluation Metric**: Accuracy

### 2. Random Forest
- **Number of Trees**: 200
- **Criterion**: Gini Impurity
- **Max Depth**: None (nodes are expanded until all leaves are pure or contain less than the minimum samples)
- **Evaluation Metric**: Accuracy

## Training and Evaluation

Both models were trained and evaluated on the MNIST dataset. Key steps included:

1. **Data Preprocessing**:
   - Normalization of pixel values (0–255) to the range (0–1).
   - Flattening the images for the Random Forest model.
   - Splitting the data into training and testing sets.

2. **Training**:
   - The ANN model was trained using the Adam optimizer, with early stopping to avoid overfitting.
   - The Random Forest model was trained on the flattened pixel data with 200 decision trees.

3. **Evaluation**:
   - Accuracy was used as the primary metric for both models, with confusion matrices generated to visualize the classification performance.

## Results

| Model              | Test Accuracy |
|--------------------|---------------|
| **ANN**            | 0.966  |
| **Random Forest**  | 0.88  |

### Visualizations
- Confusion matrices were generated to compare actual vs predicted digits.
- Sample predictions were visualized to show correctly and incorrectly classified digits.
