import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# 데이터 증가 설정
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# VGG16 모델 불러오기 (ImageNet 가중치 사용, 최상위 레이어 제외)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 사전 학습된 모델의 일부 레이어를 학습 가능하게 설정 (Fine-tuning)
for layer in base_model.layers[:15]:  
    layer.trainable = False
for layer in base_model.layers[15:]:  
    layer.trainable = True

# 새로운 모델 정의
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  
])

# 모델 컴파일 (학습률 낮추기)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 훈련과 검증 데이터를 분리하여 학습
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=30,
          validation_data=(x_test, y_test))  

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")
