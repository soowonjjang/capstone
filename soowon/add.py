from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# MediaPipe 손가락 인식 및 카메라 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# 인식된 모션을 저장할 배열
recognized_motions = []

def detect_hand_shape(hand_landmarks):
    """
    손가락 랜드마크를 기반으로 손 모양을 인식하고 숫자로 반환하는 함수.
    여기서는 간단한 예시로, 모든 손가락이 펴져 있는지 여부를 판단합니다.
    """
    # 손가락 끝과 MCP(손가락 뿌리) 랜드마크의 y 좌표를 비교하여 손가락이 펴져 있는지 판단
    extended_fingers = 0
    for i in [4, 8, 12, 16, 20]: # 손가락 끝 랜드마크 인덱스
        if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y: # 손가락 끝이 더 높으면
            extended_fingers += 1

    # 모든 손가락이 펴져 있으면 '1'로 판단, 아니면 '0'
    return 1 if extended_fingers == 5 else 0

def generate_frames():
    while True:
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 손 모양 인식 및 결과 저장
                hand_shape = detect_hand_shape(hand_landmarks)
                recognized_motions.append(hand_shape)
        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/motions')
def motions():
    """인식된 모션의 배열을 반환하는 라우트."""
    return {'recognized_motions': recognized_motions}

if __name__ == '__main__':
    app.run(debug=True, port=8080)
