const video = document.getElementById('video');
const startButton = document.getElementById('startButton');
const resultElement = document.getElementById('gestureResult');  // 제스처 결과를 표시할 HTML 요소

// 버튼 클릭 시 비디오 시작
startButton.addEventListener('click', function() {
    // 카메라 스트림 요청
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.style.display = 'block';  // 비디오가 보이도록 설정
            startButton.style.display = 'none';  // 버튼 숨기기
            startGestureRecognition();  // 수화 번역 시작
        })
        .catch(err => {
            console.error("Error accessing camera: ", err);
            alert("Error accessing camera: " + err.message);
        });
});

// 수화 번역 기능 시작
function startGestureRecognition() {
    setInterval(captureAndSendImage, 1000);  // 1초마다 이미지를 캡처하고 서버로 전송
}

function captureAndSendImage() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');

    fetch('/upload_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: dataURL })  // 이미지를 서버로 전송
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultElement.innerText = "Error: " + data.error;  // 오류 메시지 표시
        } else {
            resultElement.innerText = "Detected Gesture: " + data.gesture;  // 인식된 제스처 표시
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultElement.innerText = "Error occurred while processing the gesture.";
    });
}
