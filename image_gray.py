import cv2 as cv
import sys
img = cv.imread("./img/face1.jpeg")
if img is None:
    sys.exit("파일을 찾을수 없습니다.")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray_small= cv.resize(gray,dsize=(0,0),fx=0.5,fy=0.5)#반을 축소

cv.imwrite('soccer_gray.jpg',img)
cv.imwrite('soccer_gray_small.jpg',gray_small)

cv.imshow("Color image",img) #윈도우에 영상표시
cv.imshow("Gray image",gray) #윈도우에 영상표시
cv.imshow("Gray image small",gray_small) #윈도우에 영상표시



cv.waitKey()
cv.destroyAllWindows()