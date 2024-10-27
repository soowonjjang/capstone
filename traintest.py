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

# 'label' 열을 제외한 나머지 데이터를 처리
X = data.drop('label', axis=1).values
y = data['label'].values  # 'label' 열의 데이터를 가져옴

# 라벨 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 고유 라벨 수 및 라벨 이름 출력
num_classes = len(label_encoder.classes_)
print(f"Number of unique labels: {num_classes}")
print(f"Label names: {label_encoder.classes_}")

# 라벨 인코더 저장
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.tolist(), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 데이터 로더 설정
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 모델 및 GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size=126, num_classes=num_classes).to(device)  # 모델을 GPU로 이동
print(f"Using device: {device}")

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
for epoch in range(100):
    model.train()  # 모델을 학습 모드로 설정
    epoch_loss = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)  # 데이터를 GPU로 이동
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')

# 가중치 저장 및 출력
torch.save(model.state_dict(), 'hand_model.pth')
print('Model weights saved in hand_model.pth')

# GPU에서 사용되는지 확인
for name, param in model.named_parameters():
    if param.is_cuda:
        print(f"{name} is on GPU")
    else:
        print(f"{name} is on CPU")
print('Model training complete and saved.')