# ML Model Serving with FastAPI and Docker

A simple ML model serving API built with FastAPI and containerized with Docker.

## Features
- FastAPI REST API
- Scikit-learn model serving
- Docker containerization
- Health checks and metrics
- Logging system

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train_model.py`

## API Endpoints
- GET `/`: Root endpoint
- GET `/health`: Health check
- POST `/predict`: Make predictions
- GET `/metrics`: Get API metrics

## Technologies Used
- Python 3.9
- FastAPI
- Scikit-learn
- Docker
- Joblib
