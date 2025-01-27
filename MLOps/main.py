from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
import uvicorn
import logging
from datetime import datetime
import json
import os

# 로깅 설정
logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# FastAPI 앱 초기화
app = FastAPI(
    title="ML Model API",
    description="ML 모델을 서빙하기 위한 RESTful API",
    version="1.0.0"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_version: str
    prediction_time: str

# 모델 로드
try:
    print("\n=== DEBUG INFO ===")
    print(f"1. Current directory: {os.getcwd()}")
    print(f"2. Directory contents: {os.listdir('.')}")
    print(f"3. Model directory exists: {os.path.exists('model')}")
    
    if os.path.exists('model'):
        print(f"4. Model directory contents: {os.listdir('model')}")
        model_path = 'model/model.joblib'
        print(f"5. Full model path: {os.path.abspath(model_path)}")
        print(f"6. Model file exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            # 파일 권한 확인
            import stat
            st = os.stat(model_path)
            print(f"7. File permissions: {stat.filemode(st.st_mode)}")
            print(f"8. File size: {st.st_size} bytes")
            
            print("9. Attempting to load model...")
            model = joblib.load(model_path)
            model_version = "1.0.0"
            print("10. Model loaded successfully!")
            print(f"11. Model type: {type(model)}")
    else:
        print("Model directory not found!")
    
    print("=== END DEBUG ===\n")
    
except Exception as e:
    model = None
    print(f"\nERROR LOADING MODEL:")
    print(f"Error type: {type(e)}")
    print(f"Error message: {str(e)}")
    print(f"Error details: {e.__dict__}")
    logging.error(f"Model loading error: {e}")

# 메트릭스 저장용 변수
request_count = 0
error_count = 0
prediction_times = []

@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    request_count += 1
    logging.info(f"Path: {request.url.path} - Process Time: {process_time}s")
    
    return response

@app.get("/")
def read_root():
    return {"message": "ML Model API", "status": "active"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 상태 메트릭스 추가
    metrics = {
        "status": "healthy",
        "model_version": model_version,
        "uptime_metrics": {
            "total_requests": request_count,
            "error_rate": error_count / request_count if request_count > 0 else 0,
            "avg_prediction_time": np.mean(prediction_times) if prediction_times else 0
        }
    }
    return metrics

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global error_count
    start_time = datetime.now()
    
    try:
        # 입력 데이터를 numpy 배열로 변환
        features = np.array(request.features).reshape(1, -1)
        
        # 예측 수행
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features).max()
        
        process_time = (datetime.now() - start_time).total_seconds()
        prediction_times.append(process_time)
        
        # 예측 로깅
        logging.info(
            json.dumps({
                "prediction": float(prediction),
                "probability": float(probability),
                "features": request.features,
                "process_time": process_time
            })
        )
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_version=model_version,
            prediction_time=str(datetime.now())
        )
    
    except Exception as e:
        error_count += 1
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    return {
        "total_requests": request_count,
        "error_count": error_count,
        "average_prediction_time": np.mean(prediction_times) if prediction_times else 0,
        "model_version": model_version
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
