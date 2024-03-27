from flask import Flask, Response, render_template
import cv2
import mediapipe as mp

app = Flask(__name__)

# MediaPipe 손가락 인식 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 카메라 캡처 초기화
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, image = cap.read()
        if not success:
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # MediaPipe에 이미지 전달
        results = hands.process(image)

        # RGB 이미지를 BGR로 다시 변환하여 OpenCV에서 사용할 수 있도록 함
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 손가락 랜드마크 감지
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # 변환된 이미지를 JPEG 형식으로 인코딩
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # multipart/x-mixed-replace 형식으로 프레임 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """비디오 스트림을 표시할 메인 페이지."""
    return render_template('index.html')

@app.route('/video')
def video():
    """비디오 스트림을 제공하는 라우트."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8080)