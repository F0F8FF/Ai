import torch
import torch.nn as nn
import string
import random

# 샘플 텍스트 (더 많은 예시 추가)
text = """
The sun rises in the east, bringing warmth to the new day.
Golden rays touch morning dew, nature's diamonds on display.
Birds sing their morning song, a chorus pure and sweet.
Nature wakes anew, making each day complete.

Flowers bloom in spring time, painting gardens with delight.
Colors dance in morning sky, chasing shadows of the night.
Gentle breeze whispers soft, carrying stories through the air.
As clouds float by above, in shapes both strange and fair.

Rivers flow with stories old, through valleys deep and wide.
Mountains reach for skies of blue, standing tall with ancient pride.
Forest whispers secrets deep, in shadows cool and green.
While stars above keep watch at night, over this peaceful scene.
"""

# 텍스트를 단어로 분리하고 고유한 단어 목록 생성
words = text.split()
word_set = set(words)
word_to_ix = {word: i for i, word in enumerate(word_set)}
ix_to_word = {i: word for i, word in enumerate(word_set)}
n_words = len(word_set)

# 입력 데이터 생성
def get_input_data(text):
    words = text.split()
    return [word_to_ix[word] for word in words]

# 모델 정의
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super(WordRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embed(x)
        output, hidden = self.lstm(embedded.unsqueeze(0), hidden)
        output = self.fc(output.squeeze(0))
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

# 텍스트 생성 함수 수정
def generate_text(model, start_words=["The"], length=50):
    model.eval()
    words = start_words[:]
    hidden = model.init_hidden()

    for i in range(length):
        x = torch.tensor([word_to_ix[words[-1]]]).long()
        output, hidden = model(x, hidden)
        
        # softmax를 사용하여 확률 분포 생성
        word_weights = torch.nn.functional.softmax(output.squeeze(), dim=0)
        
        # 확률이 0 이하인 경우 처리
        if word_weights.sum().item() <= 0:
            word_weights = torch.ones_like(word_weights) / len(word_weights)
        
        # 다음 단어 선택
        word_idx = torch.multinomial(word_weights, 1)[0].item()
        next_word = ix_to_word[word_idx]
        words.append(next_word)
    
    return ' '.join(words)

# 모델 초기화 및 학습 준비
model = WordRNN(n_words)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
print("학습 시작...")
n_epochs = 1000
print_every = 100

input_data = get_input_data(text)
for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss = 0
    
    for i in range(len(input_data)-1):
        hidden = model.init_hidden()  # 매 단어마다 hidden state 초기화
        
        input_word = torch.tensor([input_data[i]]).long()
        target = torch.tensor([input_data[i+1]]).long()
        
        output, _ = model(input_word, hidden)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % print_every == 0:
        print(f'에폭: {epoch}, 평균 손실: {total_loss/len(input_data):.4f}')
        print('\n생성된 텍스트:')
        print(generate_text(model, ["The"], 30))
        print()

print("학습 완료!")

# 다양한 시작 단어로 텍스트 생성
start_words = ["The", "Golden", "Nature", "Rivers"]
print("\n=== 최종 생성된 텍스트 ===")
for start in start_words:
    print(f"\n시작 단어: '{start}'")
    print(generate_text(model, [start], 40))
