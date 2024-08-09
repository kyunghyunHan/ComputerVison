import cv2 as cv
import sys

img = cv.imread("./img/face1.jpeg")
if img is None:
    sys.exit("파일을 찾을수 없습니다.")
cv.imshow("Image Display",img) #윈도우에 영상표시
cv.waitKey()
cv.destroyAllWindows()