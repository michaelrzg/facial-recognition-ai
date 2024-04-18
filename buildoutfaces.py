print("before")
# importing the cv2 library
import cv2
from imgbeddings import imgbeddings
from PIL import Image
import numpy as np
import os
# loading the haar case algorithm file into alg variable
alg = "C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\haarcascade_frontalface_default.xml"
# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(alg)
# loading the image path into file_name variable - replace <INSERT YOUR IMAGE NAME HERE> with the path to your image
file_name = "C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\images\\gracy.jpg"
# reading the image
img = cv2.imread(file_name, 0)
# creating a black and white version of the image
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# detecting the faces
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100)
)
haar_cascade = cv2.CascadeClassifier(alg)  

imbeddings = imgbeddings()
input = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

"""
i = 105
# for each face detected

for x, y, w, h in faces:
    # crop the image to select only the face
    cropped_image = img[y : y + h, x : x + w]
    # loading the target image path into target_file_name variable  - replace <INSERT YOUR TARGET IMAGE NAME HERE> with the path to your target image
    target_file_name = 'C:\\Users\\micha\\OneDrive\\Documents\\KSU\\Projects\\Python\\xp\\outFaces\\' + str(i) + '.jpg'
    #cropped faces
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
    i = i + 1;
"""