import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands    
hands = mp_hands.Hands() # default parameters hai isme , like tracking confidence and detection confidence etc .
mpDraw = mp.solutions.drawing_utils

Ptime = 0
Ctime = 0 
 
while True:
    success , img = cap.read()
    imgrgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = hands.process(imgrgb)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                #print(id,lm)
                h , w , c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                print(id , cx , cy )
                if id == 0:
                    cv2.circle(img, (cx,cy) , 15 , (255,0,255) ,cv2.FILLED)

            mpDraw.draw_landmarks(img,handlms , mp_hands.HAND_CONNECTIONS)


    Ctime = time.time()
    fps = 1 / (Ctime-Ptime)
    Ptime = Ctime

    cv2.putText(img , str(int(fps)), (10 , 70) ,cv2.FONT_HERSHEY_COMPLEX ,3,(255,0,255),3)


    cv2.imshow("image" , img)
    cv2.waitKey(1)
    