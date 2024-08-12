import numpy as np
import cv2

# 이미지 읽기
imgL = cv2.imread('./img/body1.png', 0)
imgR = cv2.imread('./img/body2.png', 0)

# 스테레오 매칭 객체 생성 및 처리
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)

# 결과를 8비트 이미지로 변환
# OpenCV에서 disparity 이미지는 16비트로 출력되기 때문에, 8비트로 변환해줘야 합니다.
disparity_8u = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_8u = np.uint8(disparity_8u)

# 이미지 표시
cv2.imshow('Disparity', disparity_8u)
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 윈도우 닫기
ㅂㅂ