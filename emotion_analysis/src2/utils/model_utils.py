import os
import torch
import json
from datetime import datetime
import tempfile
import shutil

class ModelManager:
    def __init__(self, model, save_dir='checkpoints'):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, epoch, optimizer, loss, accuracy, filename='checkpoint.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, optimizer=None, filename='checkpoint.pth'):
        path = os.path.join(self.save_dir, filename)
        if not os.path.exists(path):
            return None
            
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
    
    def save_model(self, model, optimizer, scheduler, metrics, epoch, model_name='emotion_model'):
        """모델과 학습 상태를 안전하게 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(self.save_dir, f'{model_name}_{timestamp}')
        os.makedirs(save_path, exist_ok=True)
        
        # 임시 디렉토리에 먼저 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_path = os.path.join(temp_dir, 'model.pth')
            temp_state_path = os.path.join(temp_dir, 'training_state.pth')
            temp_config_path = os.path.join(temp_dir, 'config.json')
            
            try:
                # 모델 가중치 저장
                torch.save(
                    model.state_dict(),
                    temp_model_path,
                    _use_new_zipfile_serialization=False
                )
                
                # 학습 상태 저장
                training_state = {
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'metrics': metrics
                }
                torch.save(
                    training_state,
                    temp_state_path,
                    _use_new_zipfile_serialization=False
                )
                
                # 설정 저장
                config = {
                    'model_name': model_name,
                    'timestamp': timestamp,
                    'metrics': metrics,
                    'epoch': epoch
                }
                with open(temp_config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                
                # 임시 파일들을 최종 위치로 복사
                shutil.copy2(temp_model_path, os.path.join(save_path, 'model.pth'))
                shutil.copy2(temp_state_path, os.path.join(save_path, 'training_state.pth'))
                shutil.copy2(temp_config_path, os.path.join(save_path, 'config.json'))
                
            except Exception as e:
                print(f"모델 저장 중 오류 발생: {str(e)}")
                return None
        
        return save_path
    
    def load_model(self, model, optimizer=None, scheduler=None, model_path=None):
        """저장된 모델과 학습 상태를 로드"""
        if model_path is None:
            # 가장 최근 모델 찾기
            all_models = [d for d in os.listdir(self.save_dir) if os.path.isdir(os.path.join(self.save_dir, d))]
            if not all_models:
                raise ValueError("No saved models found")
            model_path = os.path.join(self.save_dir, sorted(all_models)[-1])
        
        try:
            # 모델 가중치 로드
            model.load_state_dict(
                torch.load(
                    os.path.join(model_path, 'model.pth'),
                    map_location=model.device
                )
            )
            
            # 학습 상태 로드
            training_state = torch.load(
                os.path.join(model_path, 'training_state.pth'),
                map_location=model.device
            )
            
            if optimizer is not None:
                optimizer.load_state_dict(training_state['optimizer_state'])
            
            if scheduler is not None and training_state['scheduler_state'] is not None:
                scheduler.load_state_dict(training_state['scheduler_state'])
            
            # 설정 로드
            with open(os.path.join(model_path, 'config.json'), 'r') as f:
                config = json.load(f)
            
            return model, optimizer, scheduler, config
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            return None
    
    def get_best_model_path(self, metric='val_accuracy', mode='max'):
        """최고 성능 모델 경로 반환"""
        best_metric = float('-inf') if mode == 'max' else float('inf')
        best_path = None
        
        for model_dir in os.listdir(self.save_dir):
            config_path = os.path.join(self.save_dir, model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                current_metric = config['metrics'].get(metric)
                if current_metric is not None:
                    if mode == 'max' and current_metric > best_metric:
                        best_metric = current_metric
                        best_path = os.path.join(self.save_dir, model_dir)
                    elif mode == 'min' and current_metric < best_metric:
                        best_metric = current_metric
                        best_path = os.path.join(self.save_dir, model_dir)
        
        return best_path 