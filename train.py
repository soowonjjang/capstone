from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from model import NeuralNet
import pickle  # 라벨 인코더 저장을 위해 필요
import os  # 파일 경로 확인을 위해 사용
import pandas as pd

# 데이터 로드
file_path = 'data.csv'  # 읽어올 파일 경로
print(f"Reading data from: {os.path.abspath(file_path)}")  # 파일 경로 출력
data = pd.read_csv(file_path)

# 이미 0으로 처리된 데이터이므로 null_to_zero 함수는 필요 없음

# 'label' 열을 제외한 나머지 데이터를 처리
X = data.drop('label', axis=1).values
y = data['label'].values  # 'label' 열의 데이터를 가져옴

# 라벨 인코딩
label_encoder = LabelEncoder()  # 라벨 인코더 객체 생성
y_encoded = label_encoder.fit_transform(y)  # 라벨 데이터를 숫자로 변환

# 고유 라벨 수 및 라벨 이름 출력
num_classes = len(label_encoder.classes_)  # 고유 라벨 수
print(f"Number of unique labels: {num_classes}")  # 고유 라벨 수 출력
print(f"Label names: {label_encoder.classes_}")  # 고유 라벨 이름 출력

# 라벨 인코더 저장
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)  # 라벨 인코더를 파일로 저장

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.tolist(), dtype=torch.float32)  # 특징 데이터를 텐서로 변환
        self.labels = torch.tensor(labels, dtype=torch.int64)  # 라벨 데이터를 텐서로 변환
    
    def __len__(self):
        return len(self.features)  # 데이터셋의 크기를 반환
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  # 인덱스에 해당하는 데이터 반환

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)  # 데이터를 훈련 세트와 테스트 세트로 분할

# 데이터 로더 설정
train_dataset = CustomDataset(X_train, y_train)  # 훈련 데이터셋 생성
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)  # 데이터 로더 생성

# 모델 생성
model = NeuralNet(input_size=126, num_classes=num_classes)  # 126개의 입력 크기와 함께 모델 생성
criterion = nn.CrossEntropyLoss()  # 손실 함수 정의
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 옵티마이저 정의

# 모델 훈련
for epoch in range(100):
    epoch_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')

# 가중치 저장 및 출력
torch.save(model.state_dict(), 'hand_model.pth')  # 훈련된 모델의 가중치를 파일로 저장
print('Model weights saved in hand_model.pth')

# 저장한 가중치의 내용 출력
for name, param in model.named_parameters():
    print(f"{name}: {param}")
print('Model training complete and saved.')
