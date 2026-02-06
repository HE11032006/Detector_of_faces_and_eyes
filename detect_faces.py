import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger les images
img = cv.imread('One_face0.jpg')

#execution de la detection de visage
# detectMultiScale(image, scaleFactor, minNeighbors)
faces = face_cascade.detectMultiScale(img, 1.05, 2)

# affichage des visages detectes
for face in faces:
    print(face)