import cv2 as cv
import numpy as np

kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(11,11))

lower_skin_color = np.array([0,10,60],dtype=np.uint8)
upper_skin_color = np.array([20,255,255],dtype=np.uint8)

mask2 = np.zeros((480,480),dtype=np.uint8)

index=0
capture = cv.VideoCapture(0)
while(True):
    key = cv.waitKey(1)
    if(key & 0xff == ord('q')):
        cv.destroyAllWindows()
        capture.release()        
        break
    elif(key & 0xff == ord('s')):
        cv.imwrite('images/Training/test/'+str(index)+'.jpg',image)
        index+=1
        print(index,end='\r')
        continue
    else:
        mask2.fill(0)
        ret,frame = capture.read()
        if(ret):
            frame = frame[0:480,0:480]
            imgHSV = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
            
            mask = cv.inRange(imgHSV,lower_skin_color,upper_skin_color)
            mask = cv.morphologyEx(mask,cv.MORPH_ERODE,kernel)
            
            blur = cv.bilateralFilter(mask,9,200,200)
            
            contours, hierarchy = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if(contours):
                contourHand = max(contours,key=lambda x : cv.contourArea(x))
                if(max):
                    cv.drawContours(mask2,[contourHand], -1, (255,255,255), -1)
            
            mask2 = cv.morphologyEx(mask2,cv.MORPH_DILATE,kernel)
            mask2 = cv.morphologyEx(mask2,cv.MORPH_DILATE,kernel)
            mask2 = cv.morphologyEx(mask2,cv.MORPH_DILATE,kernel)
            image = cv.bitwise_and(frame,frame,mask=mask2)

            cv.imshow("image",image)
            

capture.release()