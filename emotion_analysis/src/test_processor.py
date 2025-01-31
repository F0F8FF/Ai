from preprocessing.audio_processor import AudioProcessor, EmotionDataset
from preprocessing.feature_extractor import FeatureExtractor

def test_preprocessing():
    # 프로세서와 특성 추출기 초기화
    processor = AudioProcessor()
    feature_extractor = FeatureExtractor()
    
    # 데이터셋 생성
    dataset = EmotionDataset("data/raw/ravdess", processor, feature_extractor)
    
    # 첫 번째 샘플 테스트
    sample = dataset[0]
    print("\n=== 샘플 정보 ===")
    print(f"원본 오디오 shape: {sample['audio'].shape}")
    print(f"Mel Spectrogram shape: {sample['mel_spectrogram'].shape}")
    print(f"MFCC shape: {sample['mfcc'].shape}")
    print(f"감정 레이블: {sample['emotion_label']} (인덱스: {sample['emotion']})")

if __name__ == "__main__":
    test_preprocessing() 