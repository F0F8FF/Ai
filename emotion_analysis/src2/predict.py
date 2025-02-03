import torch
import librosa
import numpy as np
from models.emotion_model import DeepSeekEmotionModel
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.audio_processor import AudioProcessor

class EmotionPredictor:
    def __init__(self, model_path='checkpoints/checkpoint.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepSeekEmotionModel().to(self.device)
        
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 전처리기 초기화
        self.feature_extractor = FeatureExtractor()
        self.audio_processor = AudioProcessor(data_path="")
        
        # 감정 매핑
        self.emotion_map = {
            0: "neutral",
            1: "happy",
            2: "sad",
            3: "angry",
            4: "fearful",
            5: "disgust",
            6: "surprised",
            7: "calm"
        }
    
    def predict(self, audio_path):
        """오디오 파일에서 감정 예측"""
        try:
            # 오디오 처리
            audio = self.audio_processor.process_audio(audio_path)[0]
            features = self.feature_extractor.extract_features(audio)
            features = torch.FloatTensor(features).transpose(0, 1).unsqueeze(0)
            features = features.to(self.device)
            
            # 예측
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1).item()
                
                # 각 감정별 확률 계산
                emotion_probs = {
                    self.emotion_map[i]: round(prob.item() * 100, 2)
                    for i, prob in enumerate(probabilities[0])
                }
                
                return {
                    "predicted_emotion": self.emotion_map[predicted],
                    "confidence": round(probabilities[0][predicted].item() * 100, 2),
                    "all_probabilities": emotion_probs
                }
                
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # 예측기 초기화
    predictor = EmotionPredictor()
    
    # RAVDESS 데이터셋의 오디오 파일 경로
    test_audio = "/Users/psh/Desktop/emotion_analysis/data/RAVDESS-emotional-speech-audio/Actor_02/03-01-01-01-01-01-02.wav"
    
    # 예측
    result = predictor.predict(test_audio)
    
    # 결과 출력
    if "error" in result:
        print(f"에러 발생: {result['error']}")
    else:
        print(f"\n예측된 감정: {result['predicted_emotion']}")
        print(f"신뢰도: {result['confidence']}%")
        print("\n각 감정별 확률:")
        for emotion, prob in result['all_probabilities'].items():
            print(f"{emotion}: {prob}%") 