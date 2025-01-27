import numpy as np  # numpy 1.24.3
from sklearn.ensemble import RandomForestClassifier  # scikit-learn 1.3.0
from sklearn.datasets import make_classification
import joblib  # joblib 1.3.0
import os

# model 디렉토리 생성
if not os.path.exists('model'):
    os.makedirs('model')

# 샘플 데이터 생성
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 모델 저장
joblib.dump(model, 'model/model.joblib')
print("Model trained and saved successfully!")
