import cv2
import mediapipe as mp
import time 
import HandTrackingModule as htm 

Ptime = 0
Ctime = 0 
cap = cv2.VideoCapture(0)
detector = htm.handdetector()

while True:
    success , img = cap.read()
    img = detector.findhands(img)
    lmlist = detector.findposition(img )
    if len(lmlist) !=0 :
        print(lmlist[4])

    Ctime = time.time()
    fps = 1 / (Ctime-Ptime)
    Ptime = Ctime

    cv2.putText(img , str(int(fps)), (10 , 70) ,cv2.FONT_HERSHEY_COMPLEX ,1,(255,0,255),3)


    cv2.imshow("image" , img)
    cv2.waitKey(1)