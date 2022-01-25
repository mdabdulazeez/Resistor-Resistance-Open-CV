import cv2
import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation


'''cap = cv2.VideoCapture(0)
frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
'''
segmentor = SelfiSegmentation()
path='../sampleeeee/res1.jpg'
rescascade = cv2.CascadeClassifier("../sampleeeee/haarCascade/cascade.xml")

colors = [[(65,41,75),(114,255,255),6],
          [(0,90,40),(15,255,255),1],
          [(0,0,50),(179,50,198),8]]

def findColors(imgout,colors):
    imgHSV = cv2.cvtColor(imgout,cv2.COLOR_BGR2HSV)
    for color in colors:
        lower = np.array(color[0])
        upper = np.array(color[1])
        mask = cv2.inRange(imgHSV, lower, upper)
        #cv2.imshow(str(color[0]),mask)
        getContours(mask)


        '''pre_bil = cv2.bilateralFilter(img, 5, 80, 80)

        thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 59, 5)
        thresh = cv2.bitwise_not(thresh)

        #cv2.imshow("pre_bil", thresh)
        result = cv2.bitwise_or(mask,thresh,mask=mask)'''



        #cv2.imshow("Result",result)



def getContours(imgout):

    contours,hierarchy = cv2.findContours(imgout,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

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


    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resistor = rescascade.detectMultiScale(imgGray,1.03,4)
    for (x, y, w, h) in resistor:
        area = w*h
        if area > 200:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img,"resistor",(x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
            imgRoi = img[y:y+h,x:x+w]

            cv2.imshow("ROI", imgRoi)
    imgout = segmentor.removeBG(imgRoi, (0, 0, 0), threshold=0.4)
    imgResult = imgout.copy()

    findColors(imgout, colors)
    cv2.imshow('Result',imgResult)
    cv2.imshow("result1",img)
    cv2.imshow("result2",imgout)


    if cv2.waitKey(1) & 0xff == ord('q'):
        break
#cap.release()
cv2.destroyAllWindows()
