from cv2 import cv2
import numpy as np
import sqlite3
import os

def insertOrUpdate(Id,Name):
    conn = sqlite3.connect('/Users/Tuan/Desktop/Tutorial Python/AI/data.db')
    query = "SELECT * FROM people WHERE ID =" + str(Id)
    cursor = conn.execute(query)
    isRecordExist = 0
    for row in cursor :
        isRecordExist = 1
    if(isRecordExist == 0):
        query = "INSERT INTO people(ID,Name) values("+str(Id)+",'"+str(Name)+"')"
    else :
        query = "UPDATE people SET Name = '" + str(Name) + "' WHERE ID = " + str(Id)
    conn.execute(query)
    conn.commit()
    conn.close()

id = input("Nhap ID:")
name = input("Nhap Name:")
insertOrUpdate(id, name)

detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
sampleNum = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.imshow('Face',img) 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')
        sampleNum = sampleNum + 1
        cv2.imwrite('dataSet/User.'+str(id) +'.'+ str(sampleNum) + '.jpg', gray[y:y+h, x:x+w])
    cv2.waitKey(1)
    if sampleNum > 100 :
        break
cam.release()
cv2.destroyAllWindows()
