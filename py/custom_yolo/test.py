import cv2
import torch
from ultralytics import YOLO

# YOLO 모델 불러오기

# model = YOLO(model='yolov5s.pt') 
model = YOLO(model='./py/custom_yolo/best.pt')
results= model("./py/custom_yolo/3.jpg")
plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()
# model = YOLO(model='./weights/best2.pt')
# model = YOLO('./weights/best2.pt')
# model.export(format="onnx")
# 웹캠에서 비디오 스트림 열기
# cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 가리킵니다. 만약 다른 웹캠이나 비디오 파일을 사용하려면 경로를 지정하십시오.

# while True:
#     # 프레임 읽어오기
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLO 모델로 객체 검출 수행
#     results = model(frame)

#     # 결과 플로팅
#     plot_img = results[0].plot()
    
#     # 결과 이미지 출력
#     cv2.imshow('Object Detection', plot_img)

#     # 'q' 키를 누르면 종료
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # # 종료할 때 해제
# cap.release()
# cv2.destroyAllWindows()
# import cv2
# import torch
# from ultralytics import YOLO

# # YOLO 모델 불러오기
# model = YOLO(model='./weights/best.pt')
# scripted_model = torch.jit.script(model)
# scripted_model.save("yolo_model.pt")

# # 예제 입력 데이터 생성
# dummy_input = torch.rand(1, 256, 46, 80)  # 예제 입력 데이터 shape에 맞게 수정

# # ONNX 형식으로 모델 변환
# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
