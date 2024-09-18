#!/bin/bash

# Python3와 pip가 설치되어 있는지 확인
sudo apt update
sudo apt install -y python3 python3-pip

# 필수 pip 패키지 설치
pip3 install flask
pip3 install torch
pip3 install numpy
pip3 install opencv-python
pip3 install mediapipe
pip3 install scikit-learn
pip3 install pandas

# 추가 패키지 설치 (옵션)
pip3 install pickle5  
# pickle이 표준 라이브러리에 포함되지만 일부 환경에서 pickle5 설치가 필요할 수 있음

# 설치 확인
echo "설치된 패키지:"
pip3 list | grep -E "flask|torch|numpy|opencv-python|mediapipe|scikit-learn|pandas"