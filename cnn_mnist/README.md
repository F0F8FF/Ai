# CNN for MNIST Digit Classification

This project implements a Convolutional Neural Network (CNN) model to classify handwritten digits from the MNIST dataset. The goal is to achieve high accuracy in digit classification using TensorFlow and Keras.

## Overview

The MNIST dataset consists of 60,000 28x28 grayscale images of handwritten digits (0-9) for training, and 10,000 images for testing. In this project, we use a CNN model to classify these images with high accuracy. The model is trained using data augmentation techniques to improve generalization and prevent overfitting.

## Key Features
- **CNN Model**: Convolutional Neural Network architecture using Conv2D, MaxPooling2D, and Dense layers.
- **Data Augmentation**: Techniques like rotation, zoom, and shifting applied to training data for better model generalization.
- **Early Stopping**: Training stops when the validation loss stops improving to prevent overfitting.
