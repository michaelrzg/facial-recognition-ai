# importing the libraries
from cv2 import CascadeClassifier,VideoCapture,imwrite, cvtColor,COLOR_RGB2BGR
from imgbeddings import imgbeddings
from PIL.Image import open
import numpy as np
from numpy.linalg import norm
from time import sleep

images = []
#for filename in listdir("C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\outFaces\\"):
  #images.append(open("C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\outFaces\\"+filename))
imbeddings = imgbeddings()
#myface=[]
#for img in images:
   # myface.append(imbeddings.to_embeddings(img))
work = np.loadtxt("C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\averageVector.txt")


def euclidean_distance(a, b):
  return norm(a - b)


alg = "C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\haarcascade_frontalface_default.xml"
# passing the marhsalled algorithm to the algorithm 
haar_cascade = CascadeClassifier(alg)  
cap = VideoCapture(0)
y=500
z=312
while True:
    isItMichael=False
    sleep(1)
    ret, frame = cap.read()
    #imwrite('C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\latest_webcam_image.jpg', frame)
    gray_img = cvtColor(frame, COLOR_RGB2BGR)
    input1 = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))
    if len(input1) < 1:
       print("no face detected")
       continue
    for x, y, w, h in input1:
        # crop the image to select only the face
        cropped_image = gray_img[y : y + h, x : x + w]
        imwrite("C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\latest_processed_webcam_image.jpg",cropped_image)
          # uncomment if more data is needed
        #imwrite('C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\outFaces\\' + str(y)+'.jpg', cropped_image)
    finalImage = open("C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\latest_processed_webcam_image.jpg")
    inputface=imbeddings.to_embeddings(finalImage)
    print(euclidean_distance(inputface,work))
    if euclidean_distance(work,inputface) < 10:
       isItMichael=True
    y=y+1
    print(isItMichael)