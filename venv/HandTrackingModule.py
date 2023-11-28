import cv2
import mediapipe as mp
import time 


class handdetector():
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComp = modelComp

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findhands(self , img , draw=True ):
        imgrgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgrgb)
            #print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms , self.mpHands.HAND_CONNECTIONS) 
        
        return img
    
    def findposition(self , img , handno=0 , draw=True):
        for id,lm in enumerate(handlms.landmark):
            print(id,lm)
            h , w , c = img.shape
            cx , cy = int(lm.x*w), int(lm.y*h)
            print(id , cx , cy )
            if id == 0:
                cv2.circle(img, (cx,cy) , 15 , (255,0,255) ,cv2.FILLED)

def main():
    Ptime = 0
    Ctime = 0 
    cap = cv2.VideoCapture(0)
    detector = handdetector()
 
    while True:
        success , img = cap.read()
        img = detector.findhands(img)

        Ctime = time.time()
        fps = 1 / (Ctime-Ptime)
        Ptime = Ctime

        cv2.putText(img , str(int(fps)), (10 , 70) ,cv2.FONT_HERSHEY_COMPLEX ,3,(255,0,255),3)


        cv2.imshow("image" , img)
        cv2.waitKey(1)

if __name__ == "__main__": 
    main()