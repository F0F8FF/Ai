import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.models.emotion_model import DeepSeekEmotionModel
from src.inference.realtime_predictor import RealtimeEmotionPredictor
from src.utils.model_utils import ModelManager
import time

def main():
    try:
        # 모델 로드
        model = DeepSeekEmotionModel()
        model_manager = ModelManager()
        
        # 저장된 모델 경로 확인
        model_path = model_manager.get_best_model_path()
        if model_path is None:
            print("저장된 모델을 찾을 수 없습니다.")
            return
            
        # 모델 로드
        result = model_manager.load_model(model, model_path=model_path)
        if result is None:
            print("모델 로드에 실패했습니다.")
            return
            
        model, _, _, _ = result
        
        # 실시간 예측기 생성
        predictor = RealtimeEmotionPredictor(model)
        
        print("실시간 감정 인식 시작...")
        predictor.start()
        
        while True:
            # 최신 예측 결과 가져오기
            result = predictor.get_latest_prediction()
            if result:
                print(f"\rEmotion: {result['emotion']} (confidence: {result['confidence']:.2f})", end='')
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n종료 중...")
        if 'predictor' in locals():
            predictor.stop()
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 