import pyaudio
import numpy as np
import torch
import threading
import queue
from datetime import datetime
import librosa

class RealtimeEmotionPredictor:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 오디오 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 22050
        self.RECORD_SECONDS = 3
        
        self.frames = queue.Queue()
        self.latest_prediction = None
        self.is_running = False
        self.stream = None
        self.p = None
        self.record_thread = None
        self.predict_thread = None
        
    def start(self):
        """녹음 및 예측 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        # 녹음 스레드 시작
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.daemon = True  # 데몬 스레드로 설정
        self.record_thread.start()
        
        # 예측 스레드 시작
        self.predict_thread = threading.Thread(target=self._predict)
        self.predict_thread.daemon = True  # 데몬 스레드로 설정
        self.predict_thread.start()
    
    def stop(self):
        """녹음 및 예측 중지"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 스트림 정리
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # PyAudio 정리
        if self.p:
            self.p.terminate()
            self.p = None
        
        # 큐 비우기
        while not self.frames.empty():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
    
    def _record(self):
        """오디오 녹음"""
        while self.is_running:
            try:
                data = self.stream.read(self.CHUNK)
                self.frames.put(data)
            except Exception as e:
                print(f"녹음 중 오류 발생: {str(e)}")
    
    def _predict(self):
        """감정 예측"""
        import librosa
        
        while self.is_running:
            try:
                # 프레임 수집
                frames = []
                while len(frames) < int(self.RATE * self.RECORD_SECONDS / self.CHUNK):
                    if not self.frames.empty():
                        frames.append(self.frames.get())
                
                # 오디오 데이터 변환
                audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
                
                # 기본 정규화
                audio_data = librosa.util.normalize(audio_data)
                
                # MFCC 특성 추출 (단순화된 버전)
                mfcc = librosa.feature.mfcc(
                    y=audio_data, 
                    sr=self.RATE, 
                    n_mfcc=64,
                    n_fft=2048,
                    hop_length=512
                )
                
                # 시간 차원을 64로 조정
                target_length = 64
                if mfcc.shape[1] > target_length:
                    mfcc = mfcc[:, :target_length]
                elif mfcc.shape[1] < target_length:
                    pad_width = ((0, 0), (0, target_length - mfcc.shape[1]))
                    mfcc = np.pad(mfcc, pad_width, mode='constant')
                
                # 정규화
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
                
                # numpy 배열을 PyTorch tensor로 변환
                features = torch.FloatTensor(mfcc).unsqueeze(0)
                features = features.to(self.device)
                
                # 예측
                with torch.no_grad():
                    outputs = self.model(features)
                    
                    # 출력값 정규화
                    outputs = outputs - outputs.mean(dim=1, keepdim=True)
                    outputs = outputs / (outputs.std(dim=1, keepdim=True) + 1e-8)
                    
                    # 매우 높은 온도로 소프트맥스 적용
                    temperature = 5.0
                    scaled_outputs = outputs / temperature
                    probabilities = torch.softmax(scaled_outputs, dim=1)
                    
                    # 감정 매핑
                    emotion_map = {
                        0: "neutral",
                        1: "happy",
                        2: "sad",
                        3: "angry",
                        4: "fearful",
                        5: "disgust",
                        6: "surprised",
                        7: "calm"
                    }
                    
                    # 확률 계산 및 보정
                    raw_probs = probabilities[0].cpu().numpy()
                    
                    # 확률 보정
                    emotion_probs = {}
                    for i, emotion in emotion_map.items():
                        prob = float(raw_probs[i])
                        # disgust에 대한 페널티 적용
                        if emotion == "disgust":
                            prob *= 0.5
                        emotion_probs[emotion] = prob
                    
                    # 확률 정규화
                    total_prob = sum(emotion_probs.values())
                    emotion_probs = {k: v/total_prob for k, v in emotion_probs.items()}
                    
                    # 최소 임계값 적용
                    threshold = 0.25
                    max_prob = max(emotion_probs.values())
                    if max_prob < threshold:
                        predicted_emotion = "neutral"
                    else:
                        predicted_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
                    
                    # 결과 저장
                    self.latest_prediction = {
                        'emotion': predicted_emotion,
                        'probabilities': emotion_probs,
                        'timestamp': datetime.now()
                    }
                    
                    # 디버그 출력
                    print("\n현재 감정 확률:")
                    for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
                        print(f"{emotion}: {prob*100:.2f}%")
                    print(f"예측된 감정: {predicted_emotion}\n")
                    
            except Exception as e:
                print(f"예측 중 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def get_latest_prediction(self):
        """최신 예측 결과 반환"""
        return self.latest_prediction