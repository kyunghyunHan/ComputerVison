import cv2 as cv
import sys

img = cv.imread("./img/face1.jpeg")
if img is None:
    sys.exit("파일을 찾을수 없습니다.")

cv.rectangle(img,(830,30),(1000,200),(0,0,255),2)#직사각형 그리기
cv.putText(img,'laugh',(830,24),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

cv.imshow("Image Display",img) #윈도우에 영상표시
cv.waitKey()
cv.destroyAllWindows()