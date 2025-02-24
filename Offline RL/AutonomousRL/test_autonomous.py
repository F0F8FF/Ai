import torch
import numpy as np
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
from autonomous_dt import AutonomousDecisionTransformer

class AutonomousEvaluator:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 모델 로드
        self.model = AutonomousDecisionTransformer(
            state_dim=5,
            action_dim=2,
            max_length=50
        ).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.device = device
        
    def predict_trajectory(self, initial_state, target_return, horizon=50):
        """주어진 초기 상태에서 궤적 예측"""
        with torch.no_grad():
            # 초기 상태를 텐서로 변환
            current_state = torch.FloatTensor(initial_state).unsqueeze(0).to(self.device)
            
            # 상태와 행동 저장을 위한 numpy 배열 초기화
            states = np.zeros((horizon + 1, len(initial_state)))
            actions = np.zeros((horizon, 2))
            
            # 초기 상태 저장
            states[0] = initial_state
            
            # 목표 방향 설정
            if initial_state[3] != 0 or initial_state[4] != 0:  # 이미 움직이고 있는 경우
                target_direction = np.arctan2(initial_state[4], initial_state[3])
            else:  # 정지 상태인 경우
                target_direction = initial_state[2]  # 현재 yaw 사용
            
            # 시뮬레이션 시작
            for t in range(horizon):
                # 현재까지의 상태와 행동으로 시퀀스 생성
                state_seq = torch.FloatTensor(states[:t+1]).unsqueeze(0).to(self.device)
                if t == 0:
                    action_seq = torch.zeros((1, 1, 2)).to(self.device)
                else:
                    action_seq = torch.FloatTensor(actions[:t]).unsqueeze(0).to(self.device)
                
                # Returns-to-go 및 timesteps 설정
                returns_to_go = torch.ones((1, state_seq.shape[1])).to(self.device) * target_return
                timesteps = torch.arange(state_seq.shape[1]).to(self.device).unsqueeze(0)
                
                # 행동 예측
                action_preds = self.model(state_seq, action_seq, returns_to_go, timesteps)
                action = action_preds[0, -1].cpu().numpy()
                
                # 행동 스케일링 및 방향 조정
                action_magnitude = np.linalg.norm(action)
                if action_magnitude < 0.1:
                    action_magnitude = 0.1
                
                # 목표 방향을 향하도록 행동 조정
                action[0] = action_magnitude * np.cos(target_direction)
                action[1] = action_magnitude * np.sin(target_direction)
                
                # 행동 스케일링
                action *= 3.0
                
                # 행동 범위 제한
                action = np.clip(action, -10.0, 10.0)
                
                # 다음 상태 계산
                dt = 0.1
                next_state = states[t].copy()
                
                # 위치 업데이트
                next_state[0] += action[0] * dt
                next_state[1] += action[1] * dt
                
                # 속도 업데이트
                next_state[3] = action[0]
                next_state[4] = action[1]
                
                # yaw 업데이트 (부드러운 회전)
                current_yaw = next_state[2]
                target_yaw = np.arctan2(action[1], action[0])
                angle_diff = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))
                next_state[2] = current_yaw + 0.5 * angle_diff  # 부드러운 회전
                
                # 상태와 행동 저장
                states[t+1] = next_state
                actions[t] = action
            
            return states, actions

def visualize_trajectory(states, actions, title="Predicted Trajectory"):
    """궤적 시각화"""
    plt.figure(figsize=(10, 10))
    
    # 궤적 그리기
    plt.plot(states[:, 0], states[:, 1], 'b-', label='Vehicle Path')
    plt.scatter(states[0, 0], states[0, 1], c='g', s=100, label='Start')
    plt.scatter(states[-1, 0], states[-1, 1], c='r', s=100, label='End')
    
    # 행동 벡터 그리기 (일부만)
    for i in range(0, len(states)-1, 5):
        plt.arrow(states[i, 0], states[i, 1], 
                 actions[i, 0], actions[i, 1],
                 head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.5)
    
    plt.title(title)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory.png')
    plt.close()

def test_scenarios():
    """다양한 시나리오 테스트"""
    evaluator = AutonomousEvaluator('best_model.pth')
    
    # 테스트 시나리오들
    scenarios = [
        {
            'initial_state': [0, 0, 0, 0, 0],  # 정지 상태
            'target_return': 10.0,
            'title': 'Scenario 1: Start from rest'
        },
        {
            'initial_state': [0, 0, 0, 2, 0],  # 전진 중
            'target_return': 5.0,
            'title': 'Scenario 2: Moving forward'
        },
        {
            'initial_state': [0, 0, np.pi/4, 1, 1],  # 대각선 이동
            'target_return': 8.0,
            'title': 'Scenario 3: Diagonal movement'
        }
    ]
    
    # 각 시나리오 테스트
    for i, scenario in enumerate(scenarios):
        print(f"\nTesting {scenario['title']}")
        states, actions = evaluator.predict_trajectory(
            scenario['initial_state'],
            scenario['target_return']
        )
        
        print(f"Trajectory length: {len(states)}")
        print(f"Final position: ({states[-1, 0]:.2f}, {states[-1, 1]:.2f})")
        
        visualize_trajectory(states, actions, scenario['title'])

if __name__ == "__main__":
    test_scenarios() 