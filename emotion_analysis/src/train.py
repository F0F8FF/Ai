import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.emotion_model import DeepSeekEmotionModel
from preprocessing.audio_processor import AudioProcessor, EmotionDataset
from preprocessing.feature_extractor import FeatureExtractor
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# 토크나이저 병렬 처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EmotionTrainer:
    def __init__(self, data_path, batch_size=32, num_epochs=50, learning_rate=0.0001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # 데이터 준비
        self.prepare_data(data_path)
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-moe-16b-base",
            trust_remote_code=True
        )
        
        # 모델 초기화
        self.model = DeepSeekEmotionModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.05  # 강한 정규화
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.2,
            patience=4,
            verbose=True
        )
        
        # 텍스트 특성 미리 계산
        self.prepare_text_features()
    
    def prepare_data(self, data_path):
        dataset = EmotionDataset(data_path, AudioProcessor(), FeatureExtractor())
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # CPU에서는 멀티프로세싱 비활성화
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            num_workers=0
        )
    
    def prepare_text_features(self):
        text = "Analyze emotion in speech"
        text_encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16
        ).to(self.device)
        
        with torch.no_grad():
            self.cached_text_features = self.model.get_text_features(text_encoded)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            audio = batch['mfcc'].to(self.device)
            labels = batch['emotion'].to(self.device)
            
            outputs = self.model(audio, cached_text_features=self.cached_text_features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                audio = batch['mfcc'].to(self.device)
                labels = batch['emotion'].to(self.device)
                
                outputs = self.model(audio, cached_text_features=self.cached_text_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self):
        best_val_acc = 0
        patience = 0
        max_patience = 5  # 조기 종료를 위한 patience
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 학습률 조정
            self.scheduler.step(val_acc)
            
            # 모델 저장 및 조기 종료
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print("Early stopping!")
                    break

if __name__ == "__main__":
    trainer = EmotionTrainer("data/raw/ravdess")
    trainer.train()