# VGG16 Transfer Learning for CIFAR-10 Classification

This repository contains an implementation of transfer learning using the VGG16 model to classify images from the CIFAR-10 dataset. The model uses pre-trained weights from ImageNet, and fine-tuning is applied to adapt the model for CIFAR-10 classification.

## Overview
- **Dataset**: CIFAR-10 (32x32 RGB images of 10 different classes)
- **Model**: VGG16 (pre-trained on ImageNet)
- **Libraries Used**:
  - TensorFlow
  - Keras
  - NumPy
  - Matplotlib

## Model Architecture
The model consists of the following layers:
1. **VGG16** base model (pre-trained on ImageNet) with weights frozen in the initial layers
2. **Flatten** (to convert the 2D feature maps to 1D)
3. **Dense** (512 units, ReLU activation)
4. **Dropout** (0.5)
5. **Dense** (10 units, softmax activation) for 10-class CIFAR-10 classification

## Model Training and Evaluation
The model starts by using the pre-trained weights from ImageNet for feature extraction. The top layers of the network are replaced with new layers suited for CIFAR-10 classification. The following steps are followed:

1. Freeze the initial layers of VGG16 and add new dense layers for classification.
2. Fine-tune the last few layers of VGG16 to improve the performance.
3. Apply data augmentation to generate additional training images.
4. Compile the model with Adam optimizer and categorical cross-entropy loss.
5. Train the model for 30 epochs and evaluate the performance on the test set.

Test Accuracy: 0.82
