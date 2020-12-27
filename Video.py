import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(frame,actions = ["emotion","gender","race"])
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    str2 = result["dominant_emotion"] + "," + result["gender"] + "," + result["dominant_race"] 
    
    cv2.putText(frame,
            str2,
            (0,20),
            font,1,
            (0,0,255),
            2,
            cv2.LINE_8
            )
    
    cv2.imshow("Video",frame)
    
    if cv2.waitKey(2) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()