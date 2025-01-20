import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST 데이터셋 로드
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True, 
                                         transform=transform, 
                                         download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        transform=transform)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 입력 채널 1, 출력 채널 32, 커널 크기 3
        self.conv2 = nn.Conv2d(32, 64, 3)  # 입력 채널 32, 출력 채널 64, 커널 크기 3
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 맥스 풀링
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 완전연결층
        self.fc2 = nn.Linear(128, 10)  # 출력층 (0-9 숫자)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 첫 번째 컨볼루션 층
        x = self.pool(self.relu(self.conv2(x)))  # 두 번째 컨볼루션 층
        x = x.view(-1, 64 * 5 * 5)  # 펼치기
        x = self.relu(self.fc1(x))  # 완전연결층
        x = self.fc2(x)  # 출력층
        return x

# 모델 초기화
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
print("학습 시작...")
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[에폭: {epoch + 1}, 배치: {i + 1}] 손실: {running_loss / 100:.3f}')
            running_loss = 0.0

print("학습 완료!")

# 테스트
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'정확도: {100 * correct / total}%')
