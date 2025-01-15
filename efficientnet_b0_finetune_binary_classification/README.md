# EfficientNetB0 Fine-Tuning for Binary Classification

This project implements an image classification model using EfficientNetB0 for binary classification. The model is trained with data augmentation and fine-tuning to improve its performance on a given dataset.

## Project Overview
The model uses transfer learning with the pre-trained EfficientNetB0 architecture and fine-tuning to classify images into two categories. It leverages data augmentation techniques to enhance generalization and prevent overfitting.

### Key Features:
- **EfficientNetB0** as a base model for feature extraction.
- **Data Augmentation** to improve model robustness.
- **Fine-tuning** of the pre-trained model for better accuracy.
- **Callbacks** like EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to enhance training performance and prevent overfitting.

## Dataset
- The dataset is divided into two directories: `train` and `validation`.
- Images are resized to 224x224 pixels and normalized before training.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

Final Test accuracy: 0.97
