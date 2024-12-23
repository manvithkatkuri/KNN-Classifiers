# KNN Classifiers

This assignment focuses on applying the **K-Nearest Neighbors (KNN)** classification algorithm to the **Digits Dataset**, which contains images of hand-written digits (0 through 9). The goal is to classify these images based on their pixel intensities, evaluate the performance of the KNN classifier, and experiment with different values of `k` to optimize accuracy.

---

## Dataset Overview

The **Digits Dataset** from Scikit-learn provides 8x8 grayscale images of hand-written digits. Each image is flattened into a 64-dimensional feature vector, where each dimension represents a pixel's grayscale value. The dataset includes:
- **Features**: 64 grayscale pixel intensity values.
- **Target Labels**: Numbers 0 through 9.

---

## Objectives

1. **Visualize the Digits Dataset**: Display the first few images along with their labels.
2. **Data Preprocessing**: Split the dataset into training and testing sets for model evaluation.
3. **Train and Evaluate KNN Classifier**:
   - Train the classifier with the training dataset.
   - Evaluate its performance using metrics such as accuracy and the confusion matrix.
4. **Experiment with Different `k` Values**:
   - Test the classifier with various values of `k`.
   - Analyze how `k` impacts the model's accuracy.

---

## Steps and Key Results

### 1. Data Loading and Visualization
- The dataset is loaded using Scikit-learn's `load_digits` function.
- The first 10 images are visualized in a 2x5 grid, with labels displayed as titles.
- **Example Visualization**:
  ![Sample Image](example.png)

### 2. Data Preprocessing
- The dataset is split into training and testing sets using `train_test_split`.
- A `test_size` of 20% is used to evaluate model performance, and `random_state` ensures reproducibility.

### 3. KNN Classifier
- A **KNeighborsClassifier** is initialized with `n_neighbors=5`.
- The model is trained on the training set and predictions are made on the test set.

### 4. Evaluation
- **Accuracy**: The classifier achieved an accuracy of **X.XX%** with `k=5`.
- **Confusion Matrix**: The confusion matrix provides detailed insights into misclassifications.

### 5. Experimenting with `k` Values
- Tested `k` values: `[1, 3, 5, 7, 9, 11, 13, 15]`.
- A graph of accuracy vs. `k` values was plotted to visualize performance trends.

---

## Key Libraries and Tools

The following libraries are used:
- **Scikit-learn**:
  - `KNeighborsClassifier`: Implements the KNN algorithm.
  - `train_test_split`: Splits the dataset into training and testing sets.
  - `accuracy_score` and `confusion_matrix`: Evaluate model performance.
- **Matplotlib**: For data visualization.

---

## Instructions

1. **Set Up the Environment**:
   - Install dependencies:
     ```bash
     pip install scikit-learn matplotlib
     ```

2. **Run the Code**:
   - Execute the provided script to:
     - Visualize the dataset.
     - Train and evaluate the KNN classifier.
     - Experiment with different `k` values.

---

## Deliverables

1. **Visualization**: Subplots displaying the first 10 images and their labels.
2. **Accuracy**: Accuracy scores for each `k` value.
3. **Confusion Matrix**: Matrix showing true vs. predicted labels.
4. **Performance Graph**: Accuracy vs. `k` values plot.

---

## Results Summary

1. The KNN classifier achieves optimal accuracy with `k=X`.
2. Accuracy decreases as `k` increases beyond the optimal value, due to the smoothing effect of considering more neighbors.

---
