import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer
from src.preprocessing.feature_extractor import FeatureExtractor

class EmotionPredictor:
    def __init__(self, model, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        
        self.feature_extractor = FeatureExtractor()
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
        
        # 기본 프롬프트
        self.default_prompt = "Analyze the emotion in this speech sample."
    
    def predict(self, audio, text=None):
        """오디오에서 감정 예측
        
        Args:
            audio (np.ndarray): 오디오 신호
            text (str, optional): 분석용 프롬프트
            
        Returns:
            dict: 예측된 감정과 확률
        """
        # 특성 추출
        features = self.feature_extractor.extract_features(audio)
        features_tensor = features.unsqueeze(0).to(self.device)
        
        # 텍스트 프롬프트 설정
        if text is None:
            text = self.default_prompt
        
        # 예측
        with torch.no_grad():
            result = self.model.predict_emotion(features_tensor, text)
        
        return result 