import cv2, os
import numpy as np

from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path='dataSet'

def getImagesAndLabels(path): 
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("traning",faceNp)
        cv2.waitKey(10)
    return faces, IDs

faces, IDs = getImagesAndLabels(path)
#trainning
recognizer.train(faces, np.array(IDs))
if not os.path.exists('recognizer') :
    os.makedirs('recognizer')
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()