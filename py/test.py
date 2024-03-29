import cv2
from ultralytics import YOLO

# model = YOLO('yolov8s.pt')

model = YOLO('./weights/best.pt')
results = model('./3.jpeg') # conf=0.2, iou ..

plots = results[0].plot()
cv2.imshow("plot", plots)
cv2.waitKey(0)
cv2.destroyAllWindows()