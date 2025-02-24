from nuscenes.nuscenes import NuScenes
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class NuScenesDataset(Dataset):
    def __init__(self, dataroot, version='v1.0-mini', max_length=50):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.max_length = max_length
        
        # 더 많은 데이터 수집
        self.samples = []
        for scene in self.nusc.scene:
            sample_token = scene['first_sample_token']
            
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                self.samples.append(sample)
                sample_token = sample['next']
        
        print(f"Total samples collected: {len(self.samples)}")
        
        # 데이터 전처리
        self.trajectories = self._process_scenes()
        
    def _pad_sequence(self, sequence, max_len):
        """시퀀스를 지정된 길이로 패딩"""
        if len(sequence) > max_len:
            return sequence[:max_len]
        elif len(sequence) < max_len:
            pad_length = max_len - len(sequence)
            if len(sequence.shape) > 1:  # 2D array (states, actions)
                return np.pad(sequence, ((0, pad_length), (0, 0)), mode='constant')
            else:  # 1D array (rewards)
                return np.pad(sequence, (0, pad_length), mode='constant')
        return sequence
    
    def _process_scenes(self):
        trajectories = []
        
        for scene in self.nusc.scene:
            # 각 장면의 첫 번째 샘플
            sample_token = scene['first_sample_token']
            
            states, actions, rewards = [], [], []
            
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                
                # 라이다 데이터로부터 ego vehicle 포즈 얻기
                sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
                
                # 상태 정보 추출
                states.append([
                    ego_pose['translation'][0],  # x position
                    ego_pose['translation'][1],  # y position
                    ego_pose['rotation'][2],     # yaw
                    0,  # velocity x (not directly available)
                    0   # velocity y (not directly available)
                ])
                
                if sample['next']:
                    # 다음 상태 가져오기
                    next_sample = self.nusc.get('sample', sample['next'])
                    next_sample_data = self.nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])
                    next_pose = self.nusc.get('ego_pose', next_sample_data['ego_pose_token'])
                    
                    # 행동 계산 (현재 상태와 다음 상태의 차이)
                    action = [
                        next_pose['translation'][0] - ego_pose['translation'][0],  # dx
                        next_pose['translation'][1] - ego_pose['translation'][1]   # dy
                    ]
                    actions.append(action)
                    
                    # 보상 계산
                    reward = self._calculate_reward(ego_pose, next_pose)
                    rewards.append(reward)
                
                sample_token = sample['next']
            
            # 마지막 상태에 대한 행동과 보상 추가
            if states:
                actions.append([0, 0])  # 마지막 행동
                rewards.append(0)       # 마지막 보상
            
            if len(states) > 0:  # 빈 시퀀스 제외
                # 모든 시퀀스를 동일한 길이로 패딩
                states = self._pad_sequence(np.array(states), self.max_length)
                actions = self._pad_sequence(np.array(actions), self.max_length)
                rewards = self._pad_sequence(np.array(rewards), self.max_length)
                
                trajectories.append({
                    'states': states,
                    'actions': actions,
                    'rewards': rewards
                })
            
        return trajectories
    
    def _calculate_reward(self, current_pose, next_pose):
        """최종 개선된 보상 함수"""
        # 기본 메트릭 계산
        dx = next_pose['translation'][0] - current_pose['translation'][0]
        dy = next_pose['translation'][1] - current_pose['translation'][1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # 속도 및 방향 계산
        dt = 0.05  # 50ms
        velocity = distance / dt
        desired_direction = np.arctan2(dy, dx)
        current_yaw = current_pose['rotation'][2]
        
        # 방향 차이 계산 (더 정확한 방식)
        angle_diff = np.abs(np.arctan2(
            np.sin(desired_direction - current_yaw),
            np.cos(desired_direction - current_yaw)
        ))
        
        # 개별 보상 요소 계산
        velocity_reward = self.gaussian_reward(velocity, target=10.0, sigma=2.0)
        direction_reward = -angle_diff / np.pi
        distance_reward = self.gaussian_reward(distance, target=2.0, sigma=0.5)
        
        # 정지 상태에서의 출발 보상
        if velocity < 0.1:
            acceleration = velocity / dt
            start_reward = self.sigmoid(acceleration) * 2.0
        else:
            start_reward = 0.0
        
        # 대각선 주행 보상
        diagonal_angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
        diagonal_reward = max(self.gaussian_reward(
            angle_diff - angle, target=0, sigma=0.1
        ) for angle in diagonal_angles)
        
        # 종합 보상
        reward = (
            0.3 * velocity_reward +
            0.25 * direction_reward +
            0.2 * distance_reward +
            0.15 * start_reward +
            0.1 * diagonal_reward
        )
        
        return reward

    def gaussian_reward(self, x, target, sigma):
        """가우시안 보상 함수"""
        return np.exp(-((x - target) ** 2) / (2 * sigma ** 2))

    def sigmoid(self, x):
        """시그모이드 함수"""
        return 1 / (1 + np.exp(-x))
    
    def _augment_trajectory(self, states, actions):
        """최종 개선된 데이터 증강"""
        augmented_states = []
        augmented_actions = []
        
        # 원본 데이터
        augmented_states.append(states)
        augmented_actions.append(actions)
        
        # 8방향 회전 변환
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            rot_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            rotated_states = states.copy()
            rotated_states[:, :2] = np.dot(states[:, :2], rot_matrix.T)
            rotated_states[:, 2] = (states[:, 2] + angle) % (2*np.pi)
            rotated_actions = np.dot(actions, rot_matrix.T)
            
            augmented_states.append(rotated_states)
            augmented_actions.append(rotated_actions)
        
        # 다양한 속도 프로파일
        for scale in [0.7, 0.85, 1.15, 1.3]:
            scaled_states = states.copy()
            scaled_states[:, 3:5] *= scale
            scaled_actions = actions * scale
            
            # 가속/감속 프로파일 추가
            t = np.linspace(0, 1, len(states))
            acceleration_profile = scale * (1 + 0.2 * np.sin(2 * np.pi * t))
            
            acc_states = states.copy()
            acc_states[:, 3:5] *= acceleration_profile[:, np.newaxis]
            acc_actions = actions * acceleration_profile[:, np.newaxis]
            
            augmented_states.extend([scaled_states, acc_states])
            augmented_actions.extend([scaled_actions, acc_actions])
        
        # 정지 상태에서 출발하는 시나리오 추가
        start_states = states.copy()
        start_states[0, 3:5] = 0  # 초기 속도 0
        start_actions = actions.copy()
        start_actions[0] *= 0.5  # 부드러운 출발
        
        augmented_states.append(start_states)
        augmented_actions.append(start_actions)
        
        return augmented_states, augmented_actions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 기존 코드는 유지하고 데이터 증강 추가
        sample = self.samples[idx]
        sample_data = self._get_sample_data(sample)
        
        # 데이터 증강
        if random.random() < 0.5:  # 50% 확률로 증강
            sample_data = self._augment_data(sample_data)
        
        return sample_data
    
    def _augment_data(self, data):
        """데이터 증강"""
        states, actions, rewards, returns_to_go, timesteps = data
        
        # 무작위 회전
        angle = random.uniform(-np.pi/6, np.pi/6)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # 상태와 행동 변환
        states = states.clone()
        states[:, :2] = torch.FloatTensor(np.dot(states[:, :2].numpy(), rot_matrix.T))
        actions = torch.FloatTensor(np.dot(actions.numpy(), rot_matrix.T))
        
        # 노이즈 추가
        states += torch.randn_like(states) * 0.1
        actions += torch.randn_like(actions) * 0.05
        
        return states, actions, rewards, returns_to_go, timesteps
    
    def _get_sample_data(self, sample):
        """샘플에서 상태, 행동, 보상 데이터 추출"""
        # 샘플 시퀀스 가져오기
        sample_token = sample['token']
        states = []
        actions = []
        rewards = []
        
        while sample_token and len(states) < self.max_length:
            sample = self.nusc.get('sample', sample_token)
            
            # ego 차량의 pose 가져오기
            sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            
            # 상태 벡터 생성 [x, y, yaw, velocity_x, velocity_y]
            state = np.array([
                ego_pose['translation'][0],  # x position
                ego_pose['translation'][1],  # y position
                np.arctan2(ego_pose['rotation'][1], ego_pose['rotation'][0]),  # yaw
                ego_pose['translation'][0] - states[-1][0] if states else 0,  # velocity x
                ego_pose['translation'][1] - states[-1][1] if states else 0   # velocity y
            ])
            
            states.append(state)
            
            # 다음 샘플이 있는 경우 행동과 보상 계산
            if sample['next']:
                next_sample = self.nusc.get('sample', sample['next'])
                next_data = self.nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])
                next_pose = self.nusc.get('ego_pose', next_data['ego_pose_token'])
                
                # 행동 계산 (다음 위치로의 변화)
                action = np.array([
                    next_pose['translation'][0] - ego_pose['translation'][0],
                    next_pose['translation'][1] - ego_pose['translation'][1]
                ])
                
                # 보상 계산
                reward = self._calculate_reward(ego_pose, next_pose)
                
                actions.append(action)
                rewards.append(reward)
            
            sample_token = sample['next']
        
        # 최소 1개의 행동과 보상이 있는지 확인
        if len(actions) == 0:
            actions.append(np.zeros(2))
            rewards.append(0.0)
        
        # numpy 배열로 변환
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # 시퀀스를 정확히 max_length로 자르거나 패딩
        if len(states) > self.max_length:
            states = states[:self.max_length]
            actions = actions[:self.max_length]
            rewards = rewards[:self.max_length]
        elif len(states) < self.max_length:
            pad_length = self.max_length - len(states)
            states = np.pad(states, ((0, pad_length), (0, 0)), mode='constant')
            actions = np.pad(actions, ((0, pad_length), (0, 0)), mode='constant')
            rewards = np.pad(rewards, (0, pad_length), mode='constant')
        
        # 행동과 보상의 길이도 max_length로 맞추기
        if len(actions) > self.max_length:
            actions = actions[:self.max_length]
            rewards = rewards[:self.max_length]
        elif len(actions) < self.max_length:
            pad_length = self.max_length - len(actions)
            actions = np.pad(actions, ((0, pad_length), (0, 0)), mode='constant')
            rewards = np.pad(rewards, (0, pad_length), mode='constant')
        
        # 텐서로 변환
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        # Returns-to-go 계산
        returns_to_go = torch.cumsum(rewards.flip(0), 0).flip(0)
        
        # 시간 스텝
        timesteps = torch.arange(self.max_length)
        
        return states, actions, rewards, returns_to_go, timesteps 