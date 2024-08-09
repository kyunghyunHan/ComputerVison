import cv2 as cv
import sys

img = cv.imread("./img/face1.jpeg")
if img is None:
    sys.exit("파일을 찾을수 없습니다.")

def draw(event,x,y,flags,parm):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+200,y+200),(0,0,255),2)#직사각형 그리기
    elif event ==cv.EVENT_RBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+100,y+100),(0,0,255),2)#직사각형 그리기

    cv.imshow("Drawing",img) #윈도우에 영상표시

cv.imshow("Drawing",img) #윈도우에 영상표시
cv.setMouseCallback("Drawing",draw)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break
