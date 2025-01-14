import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# MNIST 데이터셋 로드
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 이미지 전처리: 0~255 사이의 값을 0~1 사이로 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

# 데이터 증강을 위한 생성기
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 훈련 데이터에 데이터 증강 적용
datagen.fit(train_images.reshape(-1, 28, 28, 1))  # 4D 텐서로 변환

# CNN 모델 구축
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # 2D 이미지를 3D 텐서로 변환
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# 학습률 조정
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 모델 컴파일
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# 모델 훈련
model.fit(datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=32),
          epochs=20, validation_data=(test_images, test_labels), callbacks=[early_stopping])

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
