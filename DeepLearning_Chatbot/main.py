import torch
import torch.nn as nn
import numpy as np
import json
import random

# 데이터 준비
with open('chatbot_data.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# 데이터 전처리
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = pattern.split()
        all_words.extend(w)
        xy.append((w, tag))

# 단어 전처리
all_words = [word.lower() for word in all_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# 학습 데이터 생성
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = [1 if word in pattern_sentence else 0 for word in all_words]
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# 개선된 신경망 모델
class ImprovedChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ImprovedChatbotModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)  # 추가된 레이어
        self.l4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 드롭아웃 추가
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out

# 모델 학습
input_size = len(X_train[0])
hidden_size = 16  # 히든 레이어 크기 증가
output_size = len(tags)

model = ImprovedChatbotModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 데이터를 텐서로 변환
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

# 개선된 학습 과정
num_epochs = 2000  # 에폭 수 증가
batch_size = 8

print("학습을 시작합니다...")
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 200 == 0:  # 진행상황 표시 간격 조정
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# 개선된 응답 생성 함수
def get_response(text):
    model.eval()  # 평가 모드로 설정
    
    # 입력 텍스트 전처리
    text = text.lower().split()
    
    # Bag of words 생성
    bag = [1 if word in text else 0 for word in all_words]
    X = torch.FloatTensor(bag)
    
    # 예측
    output = model(X)
    _, predicted = torch.max(output, dim=0)
    tag = tags[predicted.item()]
    
    # 확률 계산
    probs = torch.softmax(output, dim=0)
    prob = probs[predicted.item()]
    
    # 확률이 threshold보다 높은 경우에만 응답
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    
    return "죄송해요, 잘 이해하지 못했어요. 다른 방식으로 말씀해주시겠어요?"

# 개선된 대화 실행 함수
def chat():
    print("챗봇: 안녕하세요! 저와 대화를 시작해보세요. (종료하려면 'quit'를 입력하세요)")
    print("챗봇: 어떤 주제에 대해 이야기하고 싶으신가요?")
    
    while True:
        text = input("사용자: ")
        if text.lower() == 'quit':
            print("챗봇: 대화를 종료합니다. 안녕히 가세요!")
            break
            
        response = get_response(text)
        print("챗봇:", response)

if __name__ == "__main__":
    chat()
