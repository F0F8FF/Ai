FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# 필요한 패키지 설치 (버전 지정)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install numpy==1.24.3 \
    && pip install scikit-learn==1.3.0 \
    && pip install joblib==1.3.0

# 로그 디렉토리 생성
RUN mkdir logs
RUN mkdir model

# 모든 파일 복사
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]