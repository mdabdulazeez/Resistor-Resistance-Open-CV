import cv2
import numpy as np

'''cap = cv2.VideoCapture(0)
frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
'''
path='../sampleeeee/res4.jpg'

colors = [[0, 90, 40,15, 255, 255],
          [10, 70, 70,25, 255, 200],
          [23,117,170,94,255,255],
          [65,41,75,114,255,255],
          [120,54,77,179,255,219]
          ]

def findColors(img,colors):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for color in colors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        #cv2.imshow(str(color[0]),mask)
        getContours(mask)

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            cv2.drawContours(imgResult,cnt,-1,(255,0,0),3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)


while True:
    #success, img = cap.read()
    img = cv2.imread(path)
    imgResult = img.copy()
    findColors(img, colors)
    cv2.imshow('Result',imgResult)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
#cap.release()
cv2.destroyAllWindows()
