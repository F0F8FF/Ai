import sys
import os
import torch

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QProgressBar)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QIcon
import pyqtgraph as pg
import numpy as np
from datetime import datetime

from src.models.emotion_model import EmotionRecognitionModel
from src.inference.realtime_predictor import RealtimeEmotionPredictor

class EmotionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 감정 분석")
        self.setGeometry(100, 100, 1000, 600)
        
        # 메인 위젯 설정
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 상단 컨트롤 영역
        control_layout = QHBoxLayout()
        
        # 시작/정지 버튼
        self.start_button = QPushButton("시작")
        self.start_button.setFont(QFont('Arial', 12))
        self.start_button.clicked.connect(self.toggle_prediction)
        control_layout.addWidget(self.start_button)
        
        # 프롬프트 표시 레이블
        self.prompt_label = QLabel("프롬프트: -")
        self.prompt_label.setFont(QFont('Arial', 10))
        control_layout.addWidget(self.prompt_label)
        
        layout.addLayout(control_layout)
        
        # 상태 표시 레이블 (여기로 이동)
        self.status_label = QLabel("상태: 준비됨")
        self.status_label.setFont(QFont('Arial', 10))
        layout.addWidget(self.status_label)
        
        # 감정 표시 영역
        emotion_layout = QHBoxLayout()
        
        # 현재 감정 레이블
        self.emotion_label = QLabel("감정: -")
        self.emotion_label.setFont(QFont('Arial', 20))
        self.emotion_label.setAlignment(Qt.AlignCenter)
        emotion_layout.addWidget(self.emotion_label)
        
        layout.addLayout(emotion_layout)
        
        # 신뢰도 바
        confidence_layout = QHBoxLayout()
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        confidence_layout.addWidget(QLabel("신뢰도:"))
        confidence_layout.addWidget(self.confidence_bar)
        
        layout.addLayout(confidence_layout)
        
        # 그래프 영역
        self.graph = pg.PlotWidget()
        self.graph.setBackground('w')
        self.graph.setTitle("감정 변화", color="k")
        self.graph.setLabel('left', '확률', color="k")
        self.graph.setLabel('bottom', '시간 (초)', color="k")
        self.graph.showGrid(x=True, y=True)
        layout.addWidget(self.graph)
        
        # 범례 추가
        self.graph.addLegend()
        
        # 데이터 저장용 변수들
        self.times = []
        self.emotion_data = {
            "neutral": [], "happy": [], "sad": [], "angry": [],
            "fearful": [], "disgust": [], "surprised": [], "calm": []
        }
        self.curves = {}
        
        # 각 감정별 색상 설정
        colors = {
            "neutral": (100,100,100),  # 회색
            "happy": (50,205,50),      # 초록
            "sad": (30,144,255),       # 파랑
            "angry": (255,0,0),        # 빨강
            "fearful": (148,0,211),    # 보라
            "disgust": (139,69,19),    # 갈색
            "surprised": (255,215,0),   # 노랑
            "calm": (70,130,180)       # 청회색
        }
        
        for emotion, color in colors.items():
            self.curves[emotion] = self.graph.plot(pen=color, name=emotion)
        
        # 모델 초기화
        self.model = EmotionRecognitionModel()
        self.predictor = None
        self.is_running = False
        
        # 업데이트 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_prediction)
        self.timer.start(100)  # 100ms마다 업데이트
        
        # 모델 로드 (마지막으로 이동)
        self.load_model()
        
        # 종료 이벤트 연결
        self.destroyed.connect(self.cleanup)
    
    def load_model(self):
        """DeepSeek 모델 로드"""
        try:
            self.status_label.setText("상태: 모델 로딩 중...")
            
            # 모델 초기화
            self.model = EmotionRecognitionModel()
            
            # 체크포인트 경로
            checkpoint_path = 'checkpoints/checkpoint.pth'
            
            # 체크포인트 존재 확인
            if not os.path.exists(checkpoint_path):
                self.status_label.setText("상태: 체크포인트 파일이 없습니다!")
                print(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
                return
            
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # 평가 모드로 설정
            
            print(f"모델 로드 완료: {checkpoint_path}")
            print(f"체크포인트 정보:")
            print(f"- Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"- Loss: {checkpoint.get('loss', 'N/A')}")
            print(f"- Accuracy: {checkpoint.get('accuracy', 'N/A')}")
            
            self.status_label.setText("상태: 준비됨")
            self.start_button.setEnabled(True)
            
        except Exception as e:
            self.status_label.setText(f"상태: 모델 로드 실패 - {str(e)}")
            print(f"모델 로드 중 오류 발생: {str(e)}")
            self.start_button.setEnabled(False)
    
    def cleanup(self):
        """프로그램 종료 시 정리"""
        if self.predictor:
            self.predictor.stop()
            self.predictor = None
    
    def toggle_prediction(self):
        """예측 시작/정지 토글"""
        try:
            if not self.is_running:
                # 예측 시작
                if not hasattr(self, 'model') or self.model is None:
                    self.load_model()
                
                if self.predictor is None:
                    self.predictor = RealtimeEmotionPredictor(self.model)
                
                self.predictor.start()
                self.is_running = True
                self.start_button.setText("정지")
                self.status_label.setText("상태: 실행 중")
                
            else:
                # 예측 정지
                if self.predictor:
                    self.predictor.stop()
                
                self.is_running = False
                self.start_button.setText("시작")
                self.status_label.setText("상태: 준비됨")
                
                # 그래프 초기화
                for emotion_name, curve in self.curves.items():
                    curve.setData([], [])
                self.times = []
                self.emotion_data = {
                    "neutral": [], "happy": [], "sad": [], "angry": [],
                    "fearful": [], "disgust": [], "surprised": [], "calm": []
                }
                
                # 감정 레이블과 신뢰도 초기화
                self.emotion_label.setText("감정: -")
                self.confidence_bar.setValue(0)
                
        except Exception as e:
            print(f"토글 중 오류 발생: {str(e)}")
            self.status_label.setText(f"상태: 오류 발생 - {str(e)}")
    
    def update_prediction(self):
        """예측 결과 업데이트"""
        if not self.is_running or not self.predictor:
            return
            
        result = self.predictor.get_latest_prediction()
        if result:
            try:
                # 감정 레이블 업데이트
                current_emotion = result['emotion']
                probabilities = result['probabilities']
                
                # 현재 가장 높은 확률을 가진 감정 찾기
                max_prob_emotion = max(probabilities.items(), key=lambda x: x[1])
                current_emotion = max_prob_emotion[0]
                confidence = max_prob_emotion[1] * 100
                
                # UI 업데이트
                self.emotion_label.setText(f"감정: {current_emotion}")
                self.confidence_bar.setValue(int(confidence))
                
                # 상태 업데이트
                self.status_label.setText(f"상태: 분석 중 - {current_emotion} ({confidence:.1f}%)")
                
                # 그래프 데이터 업데이트
                current_time = datetime.now().timestamp()
                if not self.times:
                    self.start_time = current_time
                
                relative_time = current_time - self.start_time
                self.times.append(relative_time)
                
                # 모든 감정의 확률 업데이트
                for emotion_name, prob in probabilities.items():
                    self.emotion_data[emotion_name].append(prob)
                
                # 데이터 길이 제한 (최근 300개 포인트만 유지)
                max_points = 300
                if len(self.times) > max_points:
                    self.times = self.times[-max_points:]
                    for emotion_name in self.emotion_data:
                        self.emotion_data[emotion_name] = self.emotion_data[emotion_name][-max_points:]
                
                # 그래프 업데이트
                for emotion_name, curve in self.curves.items():
                    curve.setData(
                        self.times,
                        self.emotion_data[emotion_name]
                    )
                
            except Exception as e:
                print(f"업데이트 중 오류 발생: {str(e)}")
                self.status_label.setText(f"상태: 오류 발생 - {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = EmotionGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()