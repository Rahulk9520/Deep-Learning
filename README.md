# Deep Learning

**A collection of deep learning projects including Fashion MNIST image classification and Smart Garbage classification using TensorFlow and Keras.**  
*By Rahul Kumar*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Subprojects](#subprojects)
  - [1. Fashion MNIST Image Classification](#1-fashion-mnist-image-classification)
  - [2. Smart Garbage Classification](#2-smart-garbage-classification)
- [Directory Structure](#directory-structure)
- [Requirements & Installation](#requirements--installation)
- [Usage](#usage)
- [Results & Evaluation](#results--evaluation)
- [References](#references)
- [Contact](#contact)

---

## Project Overview

This repository contains deep learning projects focused on image classification tasks. The first subproject uses the Fashion MNIST dataset to classify clothing items, while the second tackles smart garbage classification using deep neural networks. Both projects demonstrate the application of modern deep learning techniques for real-world computer vision problems.

---

## Motivation

Deep learning has revolutionized computer vision, enabling machines to recognize and classify images with high accuracy. These projects aim to provide hands-on experience with neural network architectures, data preprocessing, model evaluation, and deployment for practical image classification tasks.

---

## Subprojects

### 1. Fashion MNIST Image Classification

- **Dataset:** Fashion MNIST (70,000 grayscale images of 10 clothing categories)
- **Objective:** Build, train, and evaluate a neural network to classify images into clothing categories.
- **Key Steps:**
  - Data loading and normalization
  - Model building (MLP and/or CNN using Keras)
  - Model training and validation
  - Performance evaluation (accuracy, loss curves, confusion matrix)
  - Visualization of predictions and misclassifications

### 2. Smart Garbage Classification

- **Dataset:** Garbage classification dataset (images of different waste categories)
- **Objective:** Classify images of garbage into categories such as plastic, paper, metal, glass, etc.
- **Key Steps:**
  - Data augmentation and preprocessing
  - Model building (CNN with transfer learning, e.g., VGG16, MobileNet)
  - Model training with data generators
  - Performance evaluation (accuracy, precision, recall, F1-score)
  - Deployment-ready model saving

---

## Directory Structure

```
Deep-Learning/
│
├── data/                                    # Datasets for both subprojects
│ ├── fashion_mnist/
│ └── garbage_classification/
├── notebooks/                               # Jupyter notebooks for EDA, modeling, and evaluation
│ ├── Fashion_MNIST_Classification.ipynb
│ └── Garbage_Classification.ipynb
├── models/                                  # Saved model weights and architectures
├── src/                                     # Source code scripts
├── results/                                 # Plots, logs, and output files
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
└── LICENSE
```

---

## Requirements & Installation

**Prerequisites:**
- Python 3.7+
- TensorFlow (>=2.x)
- Keras
- numpy, pandas
- matplotlib, seaborn
- scikit-learn
- opencv-python (for image handling)
- Pillow (for image processing)
- jupyter

**Installation:**

Clone the repository:
```bash
git clone https://github.com/YourUsername/Deep-Learning.git
cd Deep-Learning
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Fashion MNIST Image Classification

1. Open and run the notebook:
```bash
notebooks/Fashion_MNIST_Classification.ipynb
```
2. The notebook covers data loading, model training, evaluation, and visualization.

### Smart Garbage Classification

1. Prepare the garbage dataset in `data/garbage_classification/`.
2. Open and run the notebook:
```bash
notebooks/Garbage_Classification.ipynb
```
3. The notebook covers data augmentation, model building (with transfer learning), training, and evaluation.

---

## Results & Evaluation

### Fashion MNIST

- **Best Model Accuracy:** ~89% on test set.
- **Confusion Matrix:** Shows class-wise performance.

### Smart Garbage Classification

- **Best Model Accuracy:** ~92% on validation set (with transfer learning).
- **Precision, Recall, F1-score:** Evaluated for each garbage category.

---

## References

1. [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
2. [Keras Documentation](https://keras.io/)
3. [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
4. [Garbage Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
5. [scikit-learn Documentation](https://scikit-learn.org/)
6. [OpenCV Documentation](https://docs.opencv.org/)

---

## Contact

**Author:** Rahul Kumar  
**Email:** kumar.rahul226@gmail.com  
**LinkedIn:** [rk95-dataquasar](https://www.linkedin.com/in/rk95-dataquasar/)
