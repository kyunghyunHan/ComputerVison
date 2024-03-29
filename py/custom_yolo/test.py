import cv2
from ultralytics import YOLO

# YOLO 모델 불러오기
model = YOLO('./weights/best.pt')

# 웹캠에서 비디오 스트림 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 가리킵니다. 만약 다른 웹캠이나 비디오 파일을 사용하려면 경로를 지정하십시오.

while True:
    # 프레임 읽어오기
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델로 객체 검출 수행
    results = model(frame)

    # 결과 플로팅
    plot_img = results[0].plot()
    
    # 결과 이미지 출력
    cv2.imshow('Object Detection', plot_img)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료할 때 해제
cap.release()
cv2.destroyAllWindows()
