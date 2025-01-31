import os
import glob
import soundfile as sf
from collections import Counter

class DataExplorer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.emotions = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
    
    def analyze_dataset(self):
        print("데이터 분석을 시작합니다...")
        
        # 파일 수집
        search_pattern = os.path.join(self.data_path, "Actor_*", "*.wav")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"경고: {search_pattern} 에서 파일을 찾을 수 없습니다.")
            print(f"현재 디렉토리: {os.getcwd()}")
            return
        
        print(f"총 파일 수: {len(files)}")
        
        # 감정 분포 분석
        emotion_counts = []
        durations = []
        
        for file_path in files:
            try:
                # 파일명에서 감정 추출
                filename = os.path.basename(file_path)
                parts = filename.split('-')
                if len(parts) >= 3:
                    emotion = self.emotions.get(parts[2], "unknown")
                    emotion_counts.append(emotion)
                
                # 오디오 길이 계산
                audio_info = sf.info(file_path)
                durations.append(audio_info.duration)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        self._print_statistics(emotion_counts, durations)
    
    def _print_statistics(self, emotion_counts, durations):
        print("\n=== 데이터셋 통계 ===")
        
        if emotion_counts:
            print("\n감정별 샘플 수:")
            counter = Counter(emotion_counts)
            for emotion, count in counter.items():
                print(f"{emotion}: {count}")
        
        if durations:
            print("\n오디오 길이 통계:")
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            print(f"평균 길이: {avg_duration:.2f}초")
            print(f"최소 길이: {min_duration:.2f}초")
            print(f"최대 길이: {max_duration:.2f}초")

if __name__ == "__main__":
    # RAVDESS 데이터의 실제 경로 확인
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, "data", "raw", "ravdess")
    
    if not os.path.exists(data_path):
        print(f"오류: 데이터 경로를 찾을 수 없습니다: {data_path}")
        print("현재 디렉토리 구조:")
        os.system("ls -R data")
    else:
        explorer = DataExplorer(data_path)
        explorer.analyze_dataset()