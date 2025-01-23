import cv2
import numpy as np
import dlib

def apply_filter():
    # dlib의 얼굴 검출기와 특징점 검출기 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    # 필터 이미지 로드
    filter_img = cv2.imread('filter.png', cv2.IMREAD_UNCHANGED)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 검출
        faces = detector(gray)
        
        for face in faces:
            # 얼굴 특징점 검출
            landmarks = predictor(gray, face)
            
            # 눈 위치 계산
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            
            # 눈 사이 거리 계산
            eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                                 (right_eye[1] - left_eye[1])**2)
            
            # 필터 크기 조정
            filter_width = int(eye_distance * 2.5)
            filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])
            filter_resized = cv2.resize(filter_img, (filter_width, filter_height))
            
            # 필터 위치 계산 (눈 중앙)
            center_x = int((left_eye[0] + right_eye[0]) / 2)
            center_y = int((left_eye[1] + right_eye[1]) / 2)
            
            # 필터 적용 위치 계산
            x1 = center_x - filter_width // 2
            y1 = center_y - filter_height // 2
            x2 = x1 + filter_width
            y2 = y1 + filter_height
            
            # 필터가 프레임 안에 있는지 확인
            if (x1 >= 0 and y1 >= 0 and 
                x2 < frame.shape[1] and y2 < frame.shape[0]):
                # 알파 채널을 사용한 필터 합성
                alpha_channel = filter_resized[:, :, 3] / 255.0
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        frame[y1:y2, x1:x2, c] * (1 - alpha_channel) + 
                        filter_resized[:, :, c] * alpha_channel
                    )
            
            # 특징점 표시 (디버깅용)
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        cv2.imshow('Face Filter', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    apply_filter()
