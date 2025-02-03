import torch
from models.emotion_model import DeepSeekEmotionModel
from preprocessing.audio_processor import AudioProcessor, EmotionDataset
from preprocessing.feature_extractor import FeatureExtractor
from transformers import AutoTokenizer

def test_model():
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
    
    # 데이터 준비
    processor = AudioProcessor()
    feature_extractor = FeatureExtractor()
    dataset = EmotionDataset("data/raw/ravdess", processor, feature_extractor)
    
    # 모델 초기화
    model = DeepSeekEmotionModel()
    
    # 샘플 데이터로 테스트
    sample = dataset[0]
    
    # MFCC 차원 조정 (batch_size, channels, height, width)
    audio_features = sample['mfcc'].squeeze(0)  # 첫 번째 차원 제거
    if len(audio_features.shape) == 2:
        audio_features = audio_features.unsqueeze(0)  # 배치 차원 추가
    
    print(f"MFCC shape before processing: {audio_features.shape}")
    
    # 텍스트 설명 생성
    text = "This is an audio sample expressing emotion in speech"
    text_encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # 모델 실행
    with torch.no_grad():
        output = model(audio_features, text_encoded)
    
    print("\n=== DeepSeek 감정 분석 모델 테스트 ===")
    print(f"입력 오디오 특성 shape: {audio_features.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"예측된 감정 인덱스: {output.argmax().item()}")

if __name__ == "__main__":
    test_model()