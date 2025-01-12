# MNIST CNN Classification with TensorFlow

This repository contains an implementation of a Convolutional Neural Network (CNN) using TensorFlow to classify handwritten digits from the MNIST dataset.

## Overview
- **Dataset**: MNIST (28x28 grayscale images of handwritten digits)
- **Model**: Convolutional Neural Network (CNN)
- **Libraries Used**:
  - TensorFlow
  - Keras
  - scikit-learn (for data splitting)
  
## Model Architecture
The model consists of the following layers:
1. Conv2D (32 filters, 3x3 kernel)
2. MaxPooling2D (2x2 pooling)
3. Conv2D (64 filters, 3x3 kernel)
4. MaxPooling2D (2x2 pooling)
5. Flatten
6. Dense (512 units, ReLU activation)
7. Dropout (0.5)
8. Dense (10 units, softmax activation)

Test Accuracy: 0.99
