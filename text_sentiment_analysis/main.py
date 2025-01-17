import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# 1. 데이터 준비 (네이버 영화 리뷰 데이터 예시)
reviews = [
    "정말 재미있는 영화였어요",
    "시간 낭비였습니다",
    "배우들의 연기가 훌륭했습니다",
    "최악의 영화",
    # ... 더 많은 리뷰 데이터
]

labels = [1, 0, 1, 0]  # 1: 긍정, 0: 부정

# 2. 텍스트 전처리
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

# 텍스트를 숫자 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(reviews)

# 패딩 (모든 시퀀스를 동일한 길이로)
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# 3. 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 32, input_length=max_length),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 4. 모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# 5. 모델 학습
X = np.array(padded_sequences)
y = np.array(labels)

history = model.fit(X, y, 
                   epochs=10,
                   validation_split=0.2,
                   batch_size=32)

# 6. 새로운 리뷰로 예측하기
def predict_sentiment(review):
    # 새로운 리뷰 전처리
    new_sequence = tokenizer.texts_to_sequences([review])
    padded_new = pad_sequences(new_sequence, maxlen=max_length, padding='post', truncating='post')
    
    # 예측
    prediction = model.predict(padded_new)
    return "긍정" if prediction[0] > 0.5 else "부정"

# 테스트
test_review = "이 영화는 정말 재미있었고 감동적이었습니다"
print(f"리뷰: {test_review}")
print(f"감정 분석 결과: {predict_sentiment(test_review)}")

