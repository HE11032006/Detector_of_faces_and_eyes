import cv2 as cv
import mediapipe as mp

# Initialiser MediaPipe
face_detection = mp.solutions.face_detection
dessin= mp.solutions.drawing_utils

# charger les images
img = cv.imread('Two_face0.jpg')
if img is None:
    raise IOError("Impossible de charger l'image. Vérifiez le chemin d'accès.")

# Detection des visages avce mediapipe
with face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    # Convertir l'image en RGB pour MediaPipe
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        print(f"Nombre de visages détectés : {len(results.detections)}")
        for detection in results.detections:
            print(detection)
            dessin.draw_detection(img, detection)

            #afficher les coordonnées du visage
            box = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            width = int(box.width * w)
            height = int(box.height * h)
            print(f"Position : x={x}, y={y}, largeur={width}, hauteur={height}")
    else:
        print("Aucun visage détecté avec MediaPipe.")


