import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from utils.model_utils import ModelManager
from models.emotion_model import DeepSeekEmotionModel
from preprocessing.audio_processor import AudioProcessor
from preprocessing.feature_extractor import FeatureExtractor
from transformers import AutoTokenizer

def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class EmotionDataset(Dataset):
    def __init__(self, data_path, audio_processor, feature_extractor):
        self.audio_processor = audio_processor
        self.feature_extractor = feature_extractor
        
        # 오디오 파일 리스트 생성
        self.audio_files = []
        self.labels = []
        
        # .wav 파일 찾기
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(root, file))
                    # 파일 이름에서 감정 레이블 추출 (RAVDESS 형식)
                    emotion_id = int(file.split('-')[2])
                    self.labels.append(emotion_id - 1)  # 0-based indexing
        
        print(f"총 {len(self.audio_files)}개의 오디오 파일을 찾았습니다.")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # 오디오 처리
        audio = self.audio_processor.process_audio(audio_path)[0]
        features = self.feature_extractor.extract_features(audio)
        
        # 차원 순서 변경: (시간, 특성) -> (특성, 시간)
        features = torch.FloatTensor(features).transpose(0, 1)
        
        return features, torch.tensor(label, dtype=torch.long)

def train(data_path, batch_size=32, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 및 전처리기 초기화
    model = DeepSeekEmotionModel().to(device)
    audio_processor = AudioProcessor(data_path)
    feature_extractor = FeatureExtractor()
    
    # 데이터셋 및 데이터로더 설정
    dataset = EmotionDataset(data_path, audio_processor, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 모델 매니저 초기화
    model_manager = ModelManager(model)
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device), labels.to(device)
            
            # 텍스트 특징은 일단 None으로 전달
            outputs = model(features, None)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        # 체크포인트 저장
        model_manager.save_checkpoint(
            epoch, optimizer, total_loss/len(dataloader), 100.*correct/total
        )

if __name__ == "__main__":
    data_path = "경로"
    train(data_path)
