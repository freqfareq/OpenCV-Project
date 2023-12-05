import cv2 
import numpy as np
import time
import HandTrackingModule as htm 
import math 
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

############################
wcam , hcam = 1280 , 800 
############################

cap = cv2.VideoCapture(0)
cap.set(3 , wcam)
cap.set(4 , hcam) 
pTime = 0 

detector = htm.handdetector(detectionCon= 0.7) 


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()
minvol = volrange[0]
maxvol = volrange[1]
vol = 0 
volbar = 400 
volper = 0



while True:
    success , img = cap.read()
    img = detector.findhands(img)
    lmlist= detector.findposition(img , draw = False)
    if len(lmlist) != 0:
        #print(lmlist[4] , lmlist [8])

        x1 , y1 = lmlist[4][1],lmlist[4][2]
        x2 , y2 = lmlist[8][1],lmlist[8][2]
        cx , cy = (x1+x2)//2 , (y1+y2)//2

        cv2.circle(img ,(x1,y1) ,15 , (255 , 0 , 255) , cv2.FILLED)
        cv2.circle(img ,(x2,y2) ,15 , (255 , 0 , 255) , cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2) , (255,0,255), 3)
        cv2.circle(img ,(cx,cy) ,15 , (255 , 0 , 255) , cv2.FILLED)
        
        length = math.hypot(x2-x1 , y2-y1)
        #print(length)

        vol = np.interp(length,[50,300] , [minvol , maxvol])
        volbar = np.interp(length,[50,300] , [400 , 150])
        volper = np.interp(length,[50,300] , [0 , 100])
        print(int(length) , vol)
        volume.SetMasterVolumeLevel(vol , None)

        if length<50:
            cv2.circle(img ,(cx,cy) ,15 , (0 , 255 , 0) , cv2.FILLED)

    cv2.rectangle(img ,(50,150), (85,400) , (0.255,0), 3 )
    cv2.rectangle(img ,(50,int(volbar)), (85, 400) , (0,255,0), cv2.FILLED )
    cv2.putText(img, f'{int(volper)} %',(40 , 450 ) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,255,0), 3 )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(40 , 70 ) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255,0,0), 3 )

    cv2.imshow('img' , img )
    cv2.waitKey(1)
    