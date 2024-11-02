import torch
import torch.nn as nn
import torch.nn.functional as F

# 신경망 모델을 정의하는 클래스
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 200)  # 뉴런 수 증가
        self.layer2 = nn.Linear(200, 100)  # 추가 뉴런 수 증가
        self.layer3 = nn.Linear(100, 50)
        self.output = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.output(x)

# 입력 데이터가 한 손이나 양손일 때 모두 처리 가능하도록 수정합니다.
def preprocess_input(left_hand_data, right_hand_data):
    # 한 손 또는 양손 데이터를 받으면, None값을 0으로 채워서 처리합니다.
    if left_hand_data is None:
        left_hand_data = [0] * 63
    if right_hand_data is None:
        right_hand_data = [0] * 63
    return left_hand_data + right_hand_data


# 은닉층 4개 + 과적합 방지(Drop out)
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.layer1 = nn.Linear(input_size, 256)  # 첫 번째 은닉층
#         self.layer2 = nn.Linear(256, 128)         # 두 번째 은닉층
#         self.layer3 = nn.Linear(128, 64)          # 세 번째 은닉층
#         self.layer4 = nn.Linear(64, 32)           # 네 번째 은닉층
#         self.output = nn.Linear(32, num_classes)  # 출력층
#         self.dropout = nn.Dropout(0.3)            # Dropout 레이어 추가 (드롭 비율은 0.3으로 설정)

#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = self.dropout(x)                      # 첫 번째 층 뒤에 Dropout 추가
#         x = F.relu(self.layer2(x))
#         x = self.dropout(x)                      # 두 번째 층 뒤에 Dropout 추가
#         x = F.relu(self.layer3(x))
#         x = self.dropout(x)                      # 세 번째 층 뒤에 Dropout 추가
#         x = F.relu(self.layer4(x))
#         return self.output(x)

# 은닉층 5개
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.layer1 = nn.Linear(input_size, 300)  # 첫 번째 은닉층
#         self.layer2 = nn.Linear(300, 200)         # 두 번째 은닉층
#         self.layer3 = nn.Linear(200, 150)         # 세 번째 은닉층
#         self.layer4 = nn.Linear(150, 100)         # 네 번째 은닉층
#         self.layer5 = nn.Linear(100, 50)          # 다섯 번째 은닉층
#         self.output = nn.Linear(50, num_classes)  # 출력층

#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = F.relu(self.layer3(x))
#         x = F.relu(self.layer4(x))
#         x = F.relu(self.layer5(x))
#         return self.output(x)
