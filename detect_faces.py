import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger les images
img = cv.imread('Two_face0.jpg')

# Convertir en niveaux de gris
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#execution de la detection de visage
# detectMultiScale(image, scaleFactor, minNeighbors)
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

print(f"Nombre de visages détectés : {len(faces)}")

# affichage des visages detectes
for face in faces:
    print(face)

for (x, y, w, h)  in faces:
    print(f"Position : x={x}, y={y}, largeur={w}, hauteur={h}")

    # dessiner le rectangle autour du visage
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# afficher l'image principale
cv.imshow('Image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()
