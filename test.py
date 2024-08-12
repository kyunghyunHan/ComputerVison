import cv2
import numpy as np

# 초기값 설정
num_disparity = 8
block_size = 5

# StereoSGBM 객체 생성
stereo = cv2.StereoSGBM_create()

def update_num_disparity(val):
    global num_disparity
    num_disparity = val * 16
    stereo.setNumDisparities(num_disparity)
    compute_disparity()

def update_block_size(val):
    global block_size
    block_size = val
    stereo.setBlockSize(block_size)
    compute_disparity()

def compute_disparity():
    # 스테레오 이미지를 사용하여 디스파리티 계산
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # 디스파리티를 8비트로 변환하고 컬러맵 적용
    disparity = np.uint8(disp)
    disparity_color = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

    # 결과를 디스플레이
    cv2.imshow('disparity', disparity_color)

if __name__ == "__main__":
    
    imgL = cv2.imread("./img/bike0.png", cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread("./img/bike2.png", cv2.IMREAD_GRAYSCALE)

    # 윈도우 생성
    cv2.namedWindow('disparity')

    # 트랙바 생성
    cv2.createTrackbar('numDisparities', 'disparity', 0, 18, update_num_disparity)
    cv2.createTrackbar('blockSize', 'disparity', 0, 50, update_block_size)

    # 초기값 설정
    cv2.setTrackbarPos('numDisparities', 'disparity', 8)
    cv2.setTrackbarPos('blockSize', 'disparity', 5)

    compute_disparity()

    # 키 입력 대기
    cv2.waitKey(0)
    cv2.destroyAllWindows()
