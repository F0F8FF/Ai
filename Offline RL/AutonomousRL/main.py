import torch
from nuscenes_dataset import NuScenesDataset
from autonomous_dt import AutonomousDecisionTransformer
from train import AutonomousTrainer

# 개선된 모델 파라미터
MODEL_PARAMS = {
    'state_dim': 5,
    'action_dim': 2,
    'max_length': 50,
    'hidden_size': 256,  # 증가
    'n_layer': 6,       # 증가
    'n_head': 8,        # 증가
    'dropout': 0.1      # 조정
}

# 학습 파라미터 조정
TRAINING_PARAMS = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'num_epochs': 200,  # 증가
    'warmup_steps': 1000,
    'gradient_clip': 1.0
}

# Learning rate scheduler 추가
def get_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_PARAMS['num_epochs'],
        eta_min=1e-6
    )

def main():
    # 데이터셋 준비
    dataroot = "data/nuscenes"
    train_dataset = NuScenesDataset(dataroot=dataroot, version='v1.0-mini')
    
    # 모델 초기화
    model = AutonomousDecisionTransformer(
        state_dim=MODEL_PARAMS['state_dim'],
        action_dim=MODEL_PARAMS['action_dim'],
        max_length=MODEL_PARAMS['max_length'],
        hidden_size=MODEL_PARAMS['hidden_size'],
        n_layer=MODEL_PARAMS['n_layer'],
        n_head=MODEL_PARAMS['n_head'],
        dropout=MODEL_PARAMS['dropout']
    )
    
    # 학습 설정
    trainer = AutonomousTrainer(
        model=model,
        train_dataset=train_dataset,
        learning_rate=TRAINING_PARAMS['learning_rate'],
        weight_decay=TRAINING_PARAMS['weight_decay'],
        batch_size=TRAINING_PARAMS['batch_size'],
        num_epochs=TRAINING_PARAMS['num_epochs'],
        warmup_steps=TRAINING_PARAMS['warmup_steps'],
        gradient_clip=TRAINING_PARAMS['gradient_clip'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 학습 시작
    trainer.train()
    
    # 결과 시각화
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses, label='Train Loss')
    plt.plot(trainer.val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == "__main__":
    main() 
