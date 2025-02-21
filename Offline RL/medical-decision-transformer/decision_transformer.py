import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import math
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DecisionTransformerDataset(Dataset):
    def __init__(self, states, actions, rewards, returns_to_go, timesteps, max_length=20):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.rewards = torch.FloatTensor(rewards)
        self.returns_to_go = torch.FloatTensor(returns_to_go)
        self.timesteps = torch.LongTensor(timesteps)
        self.max_length = max_length

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # 시퀀스 길이가 max_length보다 길면 자르기
        states = self.states[idx][:self.max_length]
        actions = self.actions[idx][:self.max_length]
        rewards = self.rewards[idx][:self.max_length]
        returns_to_go = self.returns_to_go[idx][:self.max_length]
        timesteps = self.timesteps[idx][:self.max_length]

        return states, actions, rewards, returns_to_go, timesteps

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # 입력 텐서의 shape 확인 및 조정
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # 4D 텐서를 3D로 변환 (batch_size * something, seq_length, d_model)
        if len(x.shape) > 3:
            x = x.reshape(-1, seq_length, self.d_model)
            
        x2 = self.norm1(x)
        x2 = x2.transpose(0, 1)  # (seq_length, batch_size, d_model)
        
        # Self attention
        attn_output, _ = self.self_attn(x2, x2, x2)
        attn_output = attn_output.transpose(0, 1)
        
        # 원래 배치 크기로 복원
        if len(x.shape) > 3:
            attn_output = attn_output.reshape(batch_size, -1, seq_length, self.d_model)
            
        x = x + self.dropout1(attn_output)
        
        # Feed forward
        x2 = self.norm2(x)
        x = x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x2)))))
        
        return x

class ImprovedDecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_length=40,
        max_ep_len=1000,
        hidden_size=256,
        n_layer=4,
        n_head=4,
        n_inner=4*256,
        activation_function='relu',
        dropout=0.2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        # 향상된 임베딩
        self.token_embedding = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.position_embedding = PositionalEncoding(hidden_size, max_ep_len)

        # Transformer layers with improved architecture
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, n_head, n_inner, dropout)
            for _ in range(n_layer)
        ])

        # Improved output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, action_dim),
            nn.Tanh()
        )
        
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, state_dim)
        )
        
        self.return_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # Embed each modality
        state_embeddings = states
        action_embeddings = actions
        returns_embeddings = returns_to_go.unsqueeze(-1)

        # Concatenate modalities
        token_embeddings = torch.cat(
            [returns_embeddings, state_embeddings, action_embeddings], dim=-1
        )
        
        # Shape: [batch_size, seq_length, hidden_size]
        token_embeddings = self.token_embedding(token_embeddings)

        # Add positional embeddings
        token_embeddings = self.position_embedding(token_embeddings)

        # Pass through transformer blocks
        hidden_states = token_embeddings
        
        # Ensure correct shape
        if hidden_states.size(-1) != self.hidden_size:
            hidden_states = hidden_states.view(batch_size, seq_length, self.hidden_size)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        # Get predictions
        action_preds = self.action_head(hidden_states)  # [batch_size, seq_length, action_dim]
        state_preds = self.state_head(hidden_states)    # [batch_size, seq_length, state_dim]
        return_preds = self.return_head(hidden_states)  # [batch_size, seq_length, 1]
        
        # Print shapes for debugging
        print(f"return_preds shape before squeeze: {return_preds.shape}")
        return_preds = return_preds.squeeze(-1)  # Remove last dimension if it's 1
        print(f"return_preds shape after squeeze: {return_preds.shape}")
        print(f"returns_to_go shape: {returns_to_go.shape}")

        return action_preds, state_preds, return_preds

class ImprovedTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()  # Mixed precision training

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        metrics = {
            'action_loss': 0,
            'state_loss': 0,
            'return_loss': 0
        }
        
        for batch_idx, (states, actions, rewards, returns_to_go, timesteps) in enumerate(train_loader):
            states = states.to(self.device)
            actions = actions.to(self.device)
            returns_to_go = returns_to_go.to(self.device)
            timesteps = timesteps.to(self.device)

            # Mixed precision training
            with torch.cuda.amp.autocast():
                action_preds, state_preds, return_preds = self.model(
                    states, actions, returns_to_go, timesteps
                )

                # 텐서 크기 맞추기
                action_preds = action_preds.view(actions.shape)
                state_preds = state_preds.view(states.shape)
                return_preds = return_preds.squeeze(-1)

                # Compute losses
                action_loss = F.mse_loss(action_preds, actions)
                state_loss = F.mse_loss(state_preds, states)
                return_loss = F.mse_loss(return_preds, returns_to_go)

                loss = action_loss + state_loss + return_loss

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            total_loss += loss.item()
            metrics['action_loss'] += action_loss.item()
            metrics['state_loss'] += state_loss.item()
            metrics['return_loss'] += return_loss.item()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Average metrics
        metrics = {k: v / len(train_loader) for k, v in metrics.items()}
        metrics['total_loss'] = total_loss / len(train_loader)
        
        return metrics

    def evaluate(self, val_loader):
        self.model.eval()
        metrics = {
            'val_action_loss': 0,
            'val_state_loss': 0,
            'val_return_loss': 0,
            'val_total_loss': 0
        }
        
        with torch.no_grad():
            for states, actions, rewards, returns_to_go, timesteps in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                returns_to_go = returns_to_go.to(self.device)
                timesteps = timesteps.to(self.device)

                action_preds, state_preds, return_preds = self.model(
                    states, actions, returns_to_go, timesteps
                )

                # Compute validation losses
                action_loss = F.mse_loss(action_preds, actions)
                state_loss = F.mse_loss(state_preds, states)
                return_loss = F.mse_loss(return_preds.squeeze(-1), returns_to_go)
                total_loss = action_loss + state_loss + return_loss

                metrics['val_action_loss'] += action_loss.item()
                metrics['val_state_loss'] += state_loss.item()
                metrics['val_return_loss'] += return_loss.item()
                metrics['val_total_loss'] += total_loss.item()

        # Average metrics
        metrics = {k: v / len(val_loader) for k, v in metrics.items()}
        
        return metrics

def cross_validate(data, model_params, n_splits=5, num_epochs=50, batch_size=64):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    # 데이터셋의 원본 텐서들을 numpy 배열로 변환
    states = data.states.numpy()
    actions = data.actions.numpy()
    rewards = data.rewards.numpy()
    returns_to_go = data.returns_to_go.numpy()
    timesteps = data.timesteps.numpy()

    for fold, (train_idx, val_idx) in enumerate(kf.split(states)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # 데이터 분할
        train_dataset = DecisionTransformerDataset(
            states[train_idx],
            actions[train_idx],
            rewards[train_idx],
            returns_to_go[train_idx],
            timesteps[train_idx]
        )
        
        val_dataset = DecisionTransformerDataset(
            states[val_idx],
            actions[val_idx],
            rewards[val_idx],
            returns_to_go[val_idx],
            timesteps[val_idx]
        )
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 모델 및 옵티마이저 초기화
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImprovedDecisionTransformer(**model_params).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        
        # 학습기 초기화
        trainer = ImprovedTrainer(
            model=model,
            optimizer=optimizer,
            device=device
        )
        
        # 학습 및 평가
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader)
            
            # 모델 저장
            if val_metrics['val_total_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_total_loss']
                torch.save(model.state_dict(), f'dt_fold_{fold+1}_best.pth')
        
        fold_metrics.append({
            'fold': fold+1,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_metrics['total_loss']
        })

    return fold_metrics

class SepsisDataProcessor:
    def __init__(self):
        self.state_cols = [
            'heart_rate', 'sbp', 'dbp', 'temp', 'resp_rate',
            'wbc', 'creatinine', 'platelet', 'spo2', 'lactate'
        ]
        self.action_cols = [
            'antibiotics', 'fluid', 'vasopressor'
        ]
        self.scaler = StandardScaler()
        
    def load_mimic_data(self, path):
        """MIMIC-III 데이터 로드 및 전처리"""
        print("Loading MIMIC-III data...")
        
        # CHARTEVENTS에서 vital signs
        vitals = pd.read_csv(f"{path}/CHARTEVENTS.csv", usecols=[
            'SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUENUM'
        ])
        vitals = vitals[vitals['ITEMID'].isin([
            211,     # Heart Rate
            220045,  # Blood Pressure (systolic)
            220046,  # Blood Pressure (diastolic)
            223761,  # Temperature
            220210,  # Respiratory Rate
            220277   # SpO2
        ])]
        
        # LABEVENTS에서 lab results
        labs = pd.read_csv(f"{path}/LABEVENTS.csv", usecols=[
            'SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUENUM'
        ])
        labs = labs[labs['ITEMID'].isin([
            51301,  # WBC
            50912,  # Creatinine
            51265,  # Platelet
            50813   # Lactate
        ])]
        
        # INPUTEVENTS_MV에서 treatments
        treatments = pd.read_csv(f"{path}/INPUTEVENTS_MV.csv", usecols=[
            'SUBJECT_ID', 'HADM_ID', 'STARTTIME', 'ITEMID', 'AMOUNT', 'RATE'
        ])
        treatments = treatments[treatments['ITEMID'].isin([
            225150,  # Antibiotics
            220862,  # IV Fluids
            221906   # Vasopressors
        ])]
        
        # 패혈증 환자 식별
        sepsis = pd.read_csv(f"{path}/DIAGNOSES_ICD.csv")
        sepsis_codes = ['99591', '99592', '78552']  # 패혈증 관련 ICD-9 코드
        sepsis_patients = sepsis[sepsis['ICD9_CODE'].isin(sepsis_codes)]['SUBJECT_ID'].unique()
        
        print(f"Found {len(sepsis_patients)} sepsis patients")
        
        # 데이터 전처리
        data = self._preprocess_mimic_data(vitals, labs, treatments, sepsis_patients)
        return data
        
    def _preprocess_mimic_data(self, vitals, labs, treatments, sepsis_patients):
        """MIMIC 데이터 전처리"""
        print("Preprocessing MIMIC data...")
        
        # Vital signs 피벗
        vitals_pivot = vitals.pivot_table(
            values='VALUENUM',
            index=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME'],
            columns='ITEMID'
        ).reset_index()
        
        # Lab results 피벗
        labs_pivot = labs.pivot_table(
            values='VALUENUM',
            index=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME'],
            columns='ITEMID'
        ).reset_index()
        
        # Treatments 집계 (시간당)
        treatments['STARTTIME'] = pd.to_datetime(treatments['STARTTIME'])
        treatments_hourly = treatments.groupby(
            ['SUBJECT_ID', 'HADM_ID', pd.Grouper(key='STARTTIME', freq='H')]
        ).agg({
            'AMOUNT': 'sum',
            'RATE': 'mean'
        }).reset_index()
        
        # 데이터 병합
        data = pd.merge(
            vitals_pivot,
            labs_pivot,
            on=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME'],
            how='outer'
        )
        data = pd.merge(
            data,
            treatments_hourly,
            left_on=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME'],
            right_on=['SUBJECT_ID', 'HADM_ID', 'STARTTIME'],
            how='outer'
        )
        
        # 패혈증 환자만 필터링
        data = data[data['SUBJECT_ID'].isin(sepsis_patients)]
        
        # 결측치 처리
        data = data.sort_values(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME'])
        data = data.groupby(['SUBJECT_ID', 'HADM_ID']).ffill()
        data = data.groupby(['SUBJECT_ID', 'HADM_ID']).bfill()
        
        # 정규화
        data[self.state_cols] = self.scaler.fit_transform(data[self.state_cols])
        data[self.action_cols] = self.scaler.fit_transform(data[self.action_cols])
        
        print(f"Final dataset shape: {data.shape}")
        return data

def prepare_sequences(data, max_seq_length=20):
    """시퀀스 데이터 준비"""
    states = []
    actions = []
    rewards = []
    returns = []
    timesteps = []
    
    for patient_id in data['SUBJECT_ID'].unique():
        patient_data = data[data['SUBJECT_ID'] == patient_id]
        
        # 상태, 행동 추출
        state_seq = patient_data[state_cols].values
        action_seq = patient_data[action_cols].values
        
        # 보상 계산 (SOFA 점수 기반)
        reward_seq = -calculate_sofa_scores(patient_data)
        
        # Returns-to-go 계산
        returns_seq = np.cumsum(reward_seq[::-1])[::-1].copy()
        
        # 시퀀스 분할
        for i in range(0, len(state_seq) - max_seq_length + 1):
            states.append(state_seq[i:i+max_seq_length])
            actions.append(action_seq[i:i+max_seq_length])
            rewards.append(reward_seq[i:i+max_seq_length])
            returns.append(returns_seq[i:i+max_seq_length])
            timesteps.append(np.arange(i, i+max_seq_length))
    
    return np.array(states), np.array(actions), np.array(rewards), np.array(returns), np.array(timesteps)

def calculate_sofa_scores(patient_data):
    """SOFA 점수 계산"""
    sofa_scores = np.zeros(len(patient_data))
    
    # 심혈관계 (Mean BP or vasopressors required)
    sofa_scores += np.where(patient_data['sbp'] < 70, 4,
                  np.where(patient_data['sbp'] < 90, 3,
                  np.where(patient_data['sbp'] < 100, 2,
                  np.where(patient_data['sbp'] < 110, 1, 0))))
    
    # 호흡기계 (SpO2)
    sofa_scores += np.where(patient_data['spo2'] < 90, 4,
                  np.where(patient_data['spo2'] < 92, 3,
                  np.where(patient_data['spo2'] < 95, 2,
                  np.where(patient_data['spo2'] < 97, 1, 0))))
    
    # 혈액계 (Platelets)
    sofa_scores += np.where(patient_data['platelet'] < 20, 4,
                  np.where(patient_data['platelet'] < 50, 3,
                  np.where(patient_data['platelet'] < 100, 2,
                  np.where(patient_data['platelet'] < 150, 1, 0))))
    
    # 신장계 (Creatinine)
    sofa_scores += np.where(patient_data['creatinine'] > 5.0, 4,
                  np.where(patient_data['creatinine'] > 3.5, 3,
                  np.where(patient_data['creatinine'] > 2.0, 2,
                  np.where(patient_data['creatinine'] > 1.2, 1, 0))))
    
    # 정규화 (0-1 범위로)
    sofa_scores = sofa_scores / 16.0  # 최대 SOFA 점수는 16점
    
    return sofa_scores

class DecisionTransformerEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def predict_actions(self, states, target_return):
        """주어진 상태와 목표 리턴값에 대한 최적 행동 예측"""
        with torch.no_grad():
            # 상태 시퀀스를 텐서로 변환
            states = torch.FloatTensor(states).to(self.device)
            if len(states.shape) == 2:
                states = states.unsqueeze(0)  # batch dimension 추가
            
            # 목표 리턴값 설정
            returns_to_go = torch.ones_like(states[:, :, 0]) * target_return
            returns_to_go = returns_to_go.to(self.device)
            
            # 초기 행동 시퀀스 (0으로 초기화)
            actions = torch.zeros_like(states[:, :, :3]).to(self.device)  # 3 = action_dim
            
            # 타임스텝
            timesteps = torch.arange(states.shape[1]).long().to(self.device)
            timesteps = timesteps.unsqueeze(0).repeat(states.shape[0], 1)

            # 행동 예측
            action_preds, _, _ = self.model(states, actions, returns_to_go, timesteps)
            
            return action_preds.cpu().numpy()

    def evaluate_trajectory(self, states, actions, returns_to_go):
        """전체 궤적에 대한 모델 성능 평가"""
        metrics = {
            'action_mse': 0,
            'return_correlation': 0,
            'predicted_returns': []
        }
        
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            returns_to_go = torch.FloatTensor(returns_to_go).to(self.device)
            timesteps = torch.arange(states.shape[1]).long().to(self.device)
            timesteps = timesteps.unsqueeze(0).repeat(states.shape[0], 1)

            # 모델 예측
            action_preds, _, return_preds = self.model(states, actions, returns_to_go, timesteps)
            
            # 메트릭 계산
            metrics['action_mse'] = F.mse_loss(action_preds, actions).item()
            metrics['predicted_returns'] = return_preds.cpu().numpy()
            metrics['return_correlation'] = np.corrcoef(
                return_preds.cpu().numpy().flatten(),
                returns_to_go.cpu().numpy().flatten()
            )[0, 1]

        return metrics

def test_model_on_scenario():
    """실제 패혈증 시나리오에서 모델 테스트"""
    # 테스트 시나리오 생성
    test_states = np.array([
        # 심박수, 수축기혈압, 이완기혈압, 체온, 호흡수, WBC, 크레아티닌, 혈소판, SpO2, 젖산
        [100, 90, 60, 38.5, 22, 15, 1.8, 150, 94, 2.5],  # 초기 상태
        [105, 85, 55, 38.7, 24, 16, 2.0, 140, 92, 2.8],  # 악화
        [98, 95, 65, 38.2, 20, 14, 1.6, 160, 95, 2.2],   # 호전
    ])
    
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedDecisionTransformer(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load('dt_fold_4_best.pth'))  # 최고 성능 모델
    
    # 평가기 초기화
    evaluator = DecisionTransformerEvaluator(model, device)
    
    # 다양한 목표 리턴값에 대한 행동 예측
    target_returns = [-0.1, -0.05, 0.0]  # 다양한 목표 설정
    for target_return in target_returns:
        actions = evaluator.predict_actions(test_states, target_return)
        print(f"\nTarget return: {target_return}")
        print("Predicted actions (antibiotics, fluid, vasopressor):")
        for t, action in enumerate(actions[0]):
            print(f"Timestep {t}: {action}")

if __name__ == "__main__":
    # MIMIC-III 데이터 경로 설정
    MIMIC_PATH = "path/to/mimic-iii"
    
    # 데이터 로드 및 전처리
    processor = SepsisDataProcessor()
    data = processor.load_mimic_data(MIMIC_PATH)
    
    # 시퀀스 데이터 준비
    states, actions, rewards, returns, timesteps = prepare_sequences(data)
    
    # 데이터셋 생성
    dataset = DecisionTransformerDataset(
        states, actions, rewards, returns, timesteps
    )
    
    # 모델 학습 및 평가
    MODEL_PARAMS = {
        'state_dim': len(processor.state_cols),
        'action_dim': len(processor.action_cols),
        'max_length': 40,
        'hidden_size': 256,
        'n_layer': 4,
        'n_head': 4,
        'dropout': 0.2
    }
    
    # 교차 검증
    fold_metrics = cross_validate(
        data=dataset,
        model_params=MODEL_PARAMS,
        n_splits=5,
        num_epochs=50,
        batch_size=64
    )
    
    # 결과 출력
    print("\nCross-validation results:")
    for metrics in fold_metrics:
        print(f"Fold {metrics['fold']}:")
        print(f"  Best validation loss: {metrics['best_val_loss']:.4f}")
        print(f"  Final training loss: {metrics['final_train_loss']:.4f}")
