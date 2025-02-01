import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from utils.model_utils import ModelManager
from models.emotion_model import DeepSeekEmotionModel
from preprocessing.audio_processor import AudioProcessor, EmotionDataset
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

class EmotionTrainer:
    def __init__(self, data_path, batch_size=32, num_epochs=50, learning_rate=0.0001):
        # 시드 설정
        set_seed(42)
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 데이터 전처리
        self.audio_processor = AudioProcessor(data_path)
        self.feature_extractor = FeatureExtractor()
        
        # 데이터셋 및 데이터로더 설정
        dataset = EmotionDataset(self.audio_processor, self.feature_extractor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(42)
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # 모델 설정
        self.model = DeepSeekEmotionModel().to(self.device)
        
        # Loss 함수
        self.criterion = nn.CrossEntropyLoss()
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        self.num_epochs = num_epochs
        self.model_manager = ModelManager()
        
        # 텍스트 특성 캐싱
        self.cached_text_features = self._cache_text_features()
    
    def _cache_text_features(self):
        """텍스트 특성을 미리 계산하고 캐시"""
        text = "This is an emotional speech sample"  # 단일 텍스트 사용
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
        text_encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(text_encoded)
        return text_features  # [1, 256] 크기의 텐서 반환
    
    def train_epoch(self):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (audio, labels) in enumerate(self.train_loader):
            audio, labels = audio.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(audio, self.cached_text_features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch: {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for audio, labels in self.val_loader:
                audio, labels = audio.to(self.device), labels.to(self.device)
                outputs = self.model(audio, self.cached_text_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self):
        """전체 학습 과정"""
        best_val_acc = 0
        patience = 0
        max_patience = 10  # 조기 종료를 위한 patience
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 메트릭 저장
            metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            
            # 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                self.model_manager.save_model(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    metrics,
                    epoch
                )
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

if __name__ == "__main__":
    trainer = EmotionTrainer("data/raw/ravdess")
    trainer.train()