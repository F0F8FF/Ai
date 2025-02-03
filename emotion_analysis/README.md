# Speech Emotion Recognition Project

## Overview
A deep learning model implementation for recognizing emotions in speech using the RAVDESS dataset.

## Dataset
- Using RAVDESS dataset
- 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- File format: `03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav`

## Model Architecture
- CNN-based speech emotion recognition model
- Uses Mel-spectrogram as input
- Consists of 3 CNN layers and 2 FC layers

## Key Features
- Converts audio files to Mel-spectrograms
- CNN-based emotion classification
- Training implementation using PyTorch Lightning

## Dependencies
- Python 3.8+
- PyTorch
- PyTorch Lightning
- librosa
- numpy
- pandas

## License
MIT License

## References
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
