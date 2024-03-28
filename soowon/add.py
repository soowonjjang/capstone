# 0과 1로 출력되게 만들기

from flask import Flask, Response, render_template
import cv2
import mediapipe as mp

app = Flask(__name__)

# MediaPipe 손가락 인식 및 카메라 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, 
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0) # 손가락 함수 아직은 0과 1만 인식 되게끔
# 다 피면 1, 하나라도 접히면 0 
# 랜드마크 인덱스는 mediaPIPE에서 미리 정해둔 번호가 있음
# 4,8,12,16,20은 각 손가락의 끝 지점

# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
# 각 지점의 랜드마크 확인가능


def detect_hand_shape(hand_landmarks):
    #손가락 랜드마크를 기반으로 손 모양을 인식하고 숫자로 반환하는 함수.

    extended_fingers = 0
    for i in [4, 8, 12, 16, 20]: # 손가락 끝 랜드마크 인덱스
        if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y:
            extended_fingers += 1
    return 1 if extended_fingers == 5 else 0 #몰?루

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
                hand_shape = detect_hand_shape(hand_landmarks)
                print(f"Detected Hand Shape: {hand_shape}")  # 터미널에 출력
                
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
