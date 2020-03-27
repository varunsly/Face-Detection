import cv2


#Input Images Method
imagePath = '/home/varun/Desktop/Face_5.jpg'

#Features cascadeclassifier file path
cascadeClassifierPath = '/home/varun/Desktop/haarcascade_frontalface_alt.xml'

#Features fro the human face 
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)


image = cv2.imread(imagePath)

#Conversion of image into grayscale image 
grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Viola-Jones Algorithm 
detectedFace = cascadeClassifier.detectMultiScale(grayimage, scaleFactor = 1.1, minNeighbors= 2, minSize=(30,30))

#loop for creating face-detection windows
for(x,y,width,height) in detectedFace:
    cv2.rectangle(image,(x ,y), (x+width, y+height), (0,255,0), 5)
    
cv2.imwrite('Result-Face.jpg',image )
    