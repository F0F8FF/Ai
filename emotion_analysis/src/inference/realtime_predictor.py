import pyaudio
import numpy as np
import threading
import queue
import time
import torch
from src.inference.predictor import EmotionPredictor

class RealtimeEmotionPredictor:
    def __init__(self, model, chunk_size=2048, sample_rate=16000, record_seconds=3):
        self.predictor = EmotionPredictor(model)
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.noise_threshold = 0.005
        
        self.audio_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        self.is_running = False
        
        # 프롬프트 템플릿
        self.prompts = [
            "Analyze the emotion in this speech.",
            "What emotion is expressed in this voice?",
            "Determine the emotional state from this audio.",
            "Identify the speaker's emotion."
        ]
        self.prompt_idx = 0
        
        # PyAudio 초기화
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.lock = threading.Lock()
    
    def _get_next_prompt(self):
        """다음 프롬프트 반환"""
        prompt = self.prompts[self.prompt_idx]
        self.prompt_idx = (self.prompt_idx + 1) % len(self.prompts)
        return prompt
    
    def _process_audio(self):
        """오디오 처리 및 예측"""
        frames = []
        required_frames = int(self.record_seconds * self.sample_rate / self.chunk_size)
        
        while self.is_running:
            try:
                data = self.audio_queue.get(timeout=0.1)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # 소음 레벨 체크
                audio_level = np.abs(audio_data).mean()
                
                if audio_level > self.noise_threshold:
                    frames.append(audio_data)
                    
                    if len(frames) >= required_frames:
                        # 오디오 데이터 합치기
                        audio = np.concatenate(frames)
                        
                        try:
                            # 다양한 프롬프트로 예측
                            prompt = self._get_next_prompt()
                            result = self.predictor.predict(audio, prompt)
                            
                            # 타임스탬프 추가
                            result['timestamp'] = time.time()
                            self.result_queue.put(result)
                            
                            print(f"Emotion detected: {result['emotion']}")
                            print(f"Prompt used: {prompt}")
                            
                        except Exception as e:
                            print(f"예측 중 오류 발생: {str(e)}")
                        
                        frames = []
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"오디오 처리 중 오류 발생: {str(e)}")
    
    def start(self):
        """실시간 예측 시작"""
        self.is_running = True
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.processing_thread = threading.Thread(target=self._process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
        except Exception as e:
            print(f"오디오 스트림 시작 실패: {str(e)}")
            self.is_running = False
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 콜백 함수"""
        try:
            if not self.audio_queue.full():
                self.audio_queue.put_nowait(in_data)
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)
    
    def stop(self):
        """실시간 예측 중지"""
        with self.lock:
            self.is_running = False
            
            if self.stream is not None and self.stream.is_active():
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except OSError:
                    pass
            
            if self.p is not None:
                self.p.terminate()
            
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
    
    def get_latest_prediction(self):
        """가장 최근 예측 결과 반환"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None