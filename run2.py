from flask import Flask, render_template, Response, request, jsonify
import torch
import numpy as np
import cv2
from model import NeuralNet
import pickle
import mediapipe as mp
import time
import base64

app = Flask(__name__)

# GPU 또는 CPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델과 라벨 인코더 로드
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
model = NeuralNet(input_size=126, num_classes=num_classes).to(device)  # 모델을 GPU로 이동
model.load_state_dict(torch.load('hand_model.pth', map_location=device))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 현재 인식된 제스처와 그 제스처가 표시된 시간을 저장하는 전역 변수
current_gesture = ""
display_time = 0

# 비디오 피드를 생성하는 함수
def generate_frames():
    global current_gesture, display_time
    cap = cv2.VideoCapture(0)  # 웹캠에서 영상 캡처 시작
    prev_time = 0  # 이전 제스처 인식 시간 초기화

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        current_time = time.time()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # 이미지 좌우 반전 후 RGB로 변환
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 다시 BGR로 변환하여 출력

        left_hand_data, right_hand_data = np.zeros(63), np.zeros(63)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                if handedness.classification[0].label == 'Left':
                    left_hand_data = landmarks
                elif handedness.classification[0].label == 'Right':
                    right_hand_data = landmarks

            if current_time - prev_time > 1:  # 1초마다 수화 번역 수행
                prev_time = current_time

                input_data = np.concatenate([left_hand_data, right_hand_data])
                input_tensor = torch.tensor([input_data], dtype=torch.float32).to(device)  # 데이터도 GPU로 이동

                # 모델을 통해 제스처 예측
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    current_gesture = label_encoder.inverse_transform([predicted.item()])[0]
                    display_time = current_time

        if current_gesture and current_time - display_time < 3:
            cv2.putText(image, current_gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

        # 프레임을 클라이언트로 전송
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# 메인 페이지 라우트
@app.route('/')
def index():
    return render_template('index.html')

# 실시간 비디오 피드 라우트
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = data.get('image', '')

    # Base64 데이터를 디코딩하여 이미지를 처리
    try:
        image_data = image_data.split(',')[1]  # 'data:image/jpeg;base64,' 제거
        image = base64.b64decode(image_data)
        np_img = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            print("Image decoding failed.")
            return jsonify({'error': 'Invalid image data format'})

        print("Image successfully decoded.")

        # MediaPipe로 손 랜드마크 처리
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        left_hand_data, right_hand_data = np.zeros(63), np.zeros(63)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                if handedness.classification[0].label == 'Left':
                    left_hand_data = landmarks
                elif handedness.classification[0].label == 'Right':
                    right_hand_data = landmarks

            # 양손 데이터를 결합하여 모델 입력 데이터로 생성
            input_data = np.concatenate([left_hand_data, right_hand_data])
            input_tensor = torch.tensor([input_data], dtype=torch.float32).to(device)  # 데이터도 GPU로 이동

            # 모델을 통해 제스처 예측
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                gesture = label_encoder.inverse_transform([predicted.item()])[0]

            print(f"Detected Gesture: {gesture}")
            return jsonify({'gesture': gesture})  # 예측된 제스처 반환

        else:
            return jsonify({'gesture': 'No hand detected'})  # 손이 감지되지 않은 경우

    except Exception as e:
        print(f"Error during image processing: {str(e)}")
        return jsonify({'error': 'Image processing failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)