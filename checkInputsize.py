import torch
from model import NeuralNet

# 모델 클래스 정의
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 100)
        self.layer2 = nn.Linear(100, 50)
        self.output = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.output(x)

# 모델 로드
model = NeuralNet(input_size=63, num_classes=10)  # 임의의 값으로 초기화
model.load_state_dict(torch.load('hand_model.pth'))

# 첫 번째 레이어의 가중치 크기 확인
input_size = model.layer1.in_features
print(f"모델의 input_size: {input_size}")
