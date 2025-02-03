import sys
import pyaudio
import wave
import numpy as np
import torch
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from models.emotion_model import DeepSeekEmotionModel
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.audio_processor import AudioProcessor

class EmotionRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initModel()
        self.initAudio()
        
    def initUI(self):
        self.setWindowTitle('실시간 감정 인식')
        self.setGeometry(300, 300, 500, 300)
        
        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 감정 표시 라벨
        self.emotion_label = QLabel('감정: 대기중...', self)
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setStyleSheet('font-size: 24px; padding: 20px;')
        layout.addWidget(self.emotion_label)
        
        # 신뢰도 바
        self.confidence_bar = QProgressBar(self)
        layout.addWidget(self.confidence_bar)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        self.record_button = QPushButton('녹음 시작', self)
        self.record_button.clicked.connect(self.toggleRecording)
        button_layout.addWidget(self.record_button)
        
        layout.addLayout(button_layout)
        
        # 상태 표시줄
        self.statusBar().showMessage('준비됨')
        
        # 감정별 확률 표시 라벨들
        self.prob_labels = {}
        prob_layout = QVBoxLayout()
        for emotion in ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised", "calm"]:
            label = QLabel(f'{emotion}: 0%')
            self.prob_labels[emotion] = label
            prob_layout.addWidget(label)
        layout.addLayout(prob_layout)
        
    def initModel(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepSeekEmotionModel().to(self.device)
        
        # 모델 로드
        checkpoint = torch.load('checkpoints/checkpoint.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.feature_extractor = FeatureExtractor()
        self.audio_processor = AudioProcessor(data_path="")
        
    def initAudio(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 22050
        self.RECORD_SECONDS = 3
        
        self.p = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
        
    def toggleRecording(self):
        if not self.recording:
            self.startRecording()
        else:
            self.stopRecording()
            
    def startRecording(self):
        self.recording = True
        self.record_button.setText('녹음 중지')
        self.frames = []
        
        self.stream = self.p.open(format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                frames_per_buffer=self.CHUNK)
        
        # 녹음 스레드 시작
        self.record_thread = QThread()
        self.record_worker = RecordWorker(self.stream, self.CHUNK)
        self.record_worker.moveToThread(self.record_thread)
        
        self.record_thread.started.connect(self.record_worker.run)
        self.record_worker.frameRecorded.connect(self.processAudioFrame)
        
        self.record_thread.start()
        
    def stopRecording(self):
        self.recording = False
        self.record_button.setText('녹음 시작')
        self.stream.stop_stream()
        self.stream.close()
        self.record_thread.quit()
        self.record_thread.wait()
        
    def processAudioFrame(self, frame):
        try:
            # 오디오 데이터 처리
            audio_data = np.frombuffer(frame, dtype=np.float32)
            features = self.feature_extractor.extract_features(audio_data)
            features = torch.FloatTensor(features).transpose(0, 1).unsqueeze(0)
            features = features.to(self.device)
            
            # 감정 예측
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1).item()
                
                # 감정 매핑
                emotion_map = {
                    0: "neutral", 1: "happy", 2: "sad", 3: "angry",
                    4: "fearful", 5: "disgust", 6: "surprised", 7: "calm"
                }
                
                predicted_emotion = emotion_map[predicted]
                confidence = probabilities[0][predicted].item() * 100
                
                # UI 업데이트
                self.emotion_label.setText(f'감정: {predicted_emotion}')
                self.confidence_bar.setValue(int(confidence))
                
                # 각 감정별 확률 업데이트
                for i, emotion in emotion_map.items():
                    prob = probabilities[0][i].item() * 100
                    self.prob_labels[emotion].setText(f'{emotion}: {prob:.1f}%')
                
        except Exception as e:
            print(f"처리 중 에러 발생: {str(e)}")

class RecordWorker(QObject):
    frameRecorded = pyqtSignal(bytes)
    
    def __init__(self, stream, chunk):
        super().__init__()
        self.stream = stream
        self.chunk = chunk
        
    def run(self):
        while self.stream.is_active():
            try:
                data = self.stream.read(self.chunk)
                self.frameRecorded.emit(data)
            except Exception as e:
                print(f"녹음 중 에러 발생: {str(e)}")
                break

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionRecognitionGUI()
    ex.show()
    sys.exit(app.exec_()) 