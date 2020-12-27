import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

img1 = cv2.imread("dp.jpg")
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))

predictions1 = DeepFace.analyze(img1)


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
faces1 = faceCascade.detectMultiScale(gray1,1.3,5)

for(x,y,w,h) in faces1:
    cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))

font1 = cv2.FONT_HERSHEY_SIMPLEX
result1 = predictions1["dominant_emotion"] + "," + str(predictions1["age"]) + "," + predictions1["gender"] + "," + predictions1["dominant_race"] 

cv2.putText(frame,str2,(0,20),font,1,(0,0,255),2,cv2.LINE_8)

plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))



