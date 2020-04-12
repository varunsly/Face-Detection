import cv2
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import numpy as np

#Input Images Method
#imagePath = '/home/varun/Desktop/Face_5.jpg'

#Features cascadeclassifier file path
cascadeClassifierPath = '/home/varun/Desktop/haarcascade_frontalface_alt.xml'
eye_cascadeClassifier = '/home/varun/Desktop/haarcascade_eye.xml'


#Features fro the human face 
face_cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
eye_cascadeClassifier= cv2.CascadeClassifier(eye_cascadeClassifier)

#image = cv2.imread(imagePath)

cap=cv2.VideoCapture(-1)

while True:
    ret, image= cap.read()
    
    #Conversion of image into grayscale image 
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Viola-Jones Algorithm 
    detectedFace = face_cascadeClassifier.detectMultiScale(grayimage, scaleFactor = 1.05, minNeighbors= 2, minSize=(30,30))
    
    #loop for creating face-detection windows
    for(x,y,width,height) in detectedFace:
        cv2.rectangle(image,(x ,y), (x+width, y+height), (0,255,0), 5)
    
        e_image_gray=grayimage[y:y+height,x:x+width]
        e_image_color= image[y:y+height,x:x+width]
        
        detectedEye=eye_cascadeClassifier.detectMultiScale(e_image_gray, scaleFactor = 1.05, minNeighbors= 2, minSize=(30,30))
        
        for(ex,ey,eheight,ewidth) in detectedEye:
            cv2.rectangle(image,(ex,ey), (ex+ewidth, ey+eheight),(255,0,0),2)
            
            
    cv2.imshow('img', image)
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()

cv2.destroyAllWindows()
    



#cv2.imwrite('Result-Face.jpg',image )
    