import cv2  # OpenCV 라이브러리 임포트
import mediapipe as mp  # MediaPipe 라이브러리 임포트
import csv  # CSV 파일을 다루기 위한 라이브러리 임포트
import time  # 시간 관련 함수 사용을 위한 라이브러리 임포트
import os  # 파일 존재 여부 확인을 위한 라이브러리

# MediaPipe의 손 인식 솔루션 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # 양손을 감지하도록 수정
    min_detection_confidence=0.5,  # 최소 감지 신뢰도
    min_tracking_confidence=0.5  # 최소 추적 신뢰도
)
mp_draw = mp.solutions.drawing_utils  # 랜드마크를 그리기 위한 도구 초기화

# 웹캠에서 비디오 캡처 시작
cap = cv2.VideoCapture(0)

# 카메라가 제대로 열렸는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 파일 작성 모드를 선택
mode = input("데이터를 추가할 것인지(a), 새로운 파일을 만들 것인지(n)를 입력하세요: ")

# 파일 경로와 모드 결정
file_exists = os.path.exists('data.csv')

if mode == 'n':
    # 새로운 파일을 생성하는 경우
    file_mode = 'w'
    print("새로운 CSV 파일을 생성합니다.")
elif mode == 'a' and file_exists:
    # 파일이 존재할 때 데이터를 추가
    file_mode = 'a'
    print("기존 CSV 파일에 데이터를 추가합니다.")
else:
    print("유효한 선택이 아닙니다. 프로그램을 종료합니다.")
    exit()

# CSV 파일 열기 및 헤더 작성
with open('data.csv', file_mode, newline='') as file:
    csv_writer = csv.writer(file)
    
    # 새로운 파일을 생성하는 경우에만 헤더 작성
    if file_mode == 'w':
        headers = ['label']
        for i in range(21):  # 왼손 랜드마크
            headers.extend([f'left_x{i}', f'left_y{i}', f'left_z{i}'])
        for i in range(21):  # 오른손 랜드마크
            headers.extend([f'right_x{i}', f'right_y{i}', f'right_z{i}'])
        csv_writer.writerow(headers)

    # 원하는 레이블의 수 입력받기
    num_labels = int(input("원하시는 레이블의 수(수화의 개수)를 입력하세요: "))

    # 레이블 입력 및 데이터 수집 반복
    for _ in range(num_labels):  # 입력된 레이블 수만큼 반복
        input("준비가 완료되면 아무키나 입력하세요: ")  # 새로운 레이블 수집 시작
        label = input("레이블을 입력하세요: ")  # 레이블 이름 입력

        count = 0
        while count < 1000:  # 각 레이블에 대해 1000개의 데이터 수집
            success, image = cap.read()  # 웹캠에서 이미지 읽기
            if not success:  # 이미지 읽기에 실패하면 다음 반복으로 넘어감
                continue

            image = cv2.flip(image, 1)  # 이미지 좌우 반전 (거울 모드)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지를 RGB로 변환
            results = hands.process(image)  # 손 랜드마크 감지
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 이미지를 다시 BGR로 변환 (OpenCV 디스플레이 용)

            if results.multi_hand_landmarks:  # 손 랜드마크가 감지된 경우
                row = [label]  # 현재 레이블 추가
                left_hand_data = [0] * 63  # 왼손 랜드마크 자리 초기화 (0으로 대체)
                right_hand_data = [0] * 63  # 오른손 랜드마크 자리 초기화 (0으로 대체)
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = handedness.classification[0].label  # 왼손/오른손 구분

                    # 각 랜드마크의 x, y, z 좌표를 row에 추가
                    if hand_type == 'Left':
                        left_hand_data = []
                        for lm in hand_landmarks.landmark[:21]:
                            left_hand_data.extend([lm.x, lm.y, lm.z])
                    elif hand_type == 'Right':
                        right_hand_data = []
                        for lm in hand_landmarks.landmark[:21]:
                            right_hand_data.extend([lm.x, lm.y, lm.z])

                    # 랜드마크 그리기
                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 감지되지 않은 손에 대해서는 0으로 대체된 left_hand_data 또는 right_hand_data 사용
                row.extend(left_hand_data + right_hand_data)
                
                csv_writer.writerow(row)
                count += 1  # 수집한 데이터 수 증가
                time.sleep(0.01)  # 0.05초 대기 (0.05초마다 한 번씩 데이터 기록)

            # 이미지 화면에 표시
            cv2.imshow("Hand Tracking", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 반복 종료
                break

        print(f"레이블 {label}에 대해 1000개의 데이터를 수집했습니다.")  # 레이블에 대한 데이터 수집 완료 메시지 출력

# 자원 해제
cap.release()  # 웹캠 자원 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
