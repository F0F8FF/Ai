from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from pydantic import BaseModel
import time
import numpy as np
from datetime import datetime
import logging
import joblib

# 모델 로드
try:
    model = joblib.load('model/model.joblib')
    model_version = "1.0.0"
    logging.info(f"Model loaded successfully. Version: {model_version}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None
    model_version = "unknown"

# Pydantic 모델 정의
class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_version: str
    prediction_time: str

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 메트릭스 정의
REQUESTS = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
LATENCY = Histogram('api_latency_seconds', 'API latency', ['method', 'endpoint'])
PREDICTIONS = Counter('model_predictions_total', 'Total model predictions')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Model prediction latency')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    start_time = time.time()
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features).max()
        
        duration = time.time() - start_time
        PREDICTIONS.inc()
        PREDICTION_LATENCY.observe(duration)
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version=model_version,
            prediction_time=str(datetime.now())
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
