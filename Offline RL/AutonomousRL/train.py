import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from nuscenes_dataset import NuScenesDataset
from autonomous_dt import AutonomousDecisionTransformer

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch in progress_bar:
        # 데이터를 GPU로 이동
        states, actions, rewards, returns_to_go, timesteps = [b.to(device) for b in batch]
        
        # Forward pass
        action_preds = model(states, actions, returns_to_go, timesteps)
        loss = F.mse_loss(action_preds, actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 그래디언트 클리핑
        optimizer.step()
        
        # 손실 기록
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            states, actions, rewards, returns_to_go, timesteps = [b.to(device) for b in batch]
            action_preds = model(states, actions, returns_to_go, timesteps)
            loss = F.mse_loss(action_preds, actions)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 하이퍼파라미터
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    NUM_HEADS = 4
    DROPOUT = 0.1
    
    # 데이터셋 로드
    print("Loading dataset...")
    dataset = NuScenesDataset(
        dataroot='data/nuscenes',
        version='v1.0-mini',
        max_length=50
    )
    
    # 데이터셋 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드 설정
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True  # GPU 전송 속도 향상
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of batches per epoch: {len(train_loader)}")
    
    # 모델 초기화
    model = AutonomousDecisionTransformer(
        state_dim=5,
        action_dim=2,
        max_length=50,
        hidden_size=HIDDEN_SIZE,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01  # L2 정규화
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 학습 기록용
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # 학습 루프
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # 학습
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # 검증
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # 학습률 조정
        scheduler.step(val_loss)
        
        # 진행상황 출력
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # 모델 저장 (검증 손실이 개선된 경우)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model!")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 조기 종료
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs!")
            break
    
    # 학습 곡선 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.close()
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()