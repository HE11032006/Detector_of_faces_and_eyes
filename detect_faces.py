import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'detector.tflite'
IMAGE_PATH = 'One_face0.jpg'

# 1. Configuration du détecteur
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)

with vision.FaceDetector.create_from_options(options) as detector:
    # Charger l'image
    img = cv.imread(IMAGE_PATH)
    if img is None:
        raise IOError(f"Impossible de charger l'image {IMAGE_PATH}")

    # Convertir pour MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    )

    # Détecter
    result = detector.detect(mp_image)

    # Compter les visages
    num_faces = len(result.detections)
    if num_faces > 1:
        print(f"ALERTE : {num_faces} visages détectés !")
    elif num_faces == 0:
        print("ALERTE : Aucun visage visible !")
    else:
        print("Un seul visage détecté. Tout est OK.")

    # Dessiner les résultats pour vérifier
    for detection in result.detections:
        bbox = detection.bounding_box
        start_point = int(bbox.origin_x), int(bbox.origin_y)
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)

        # Dessiner le rectangle (Vert si OK, Rouge si plusieurs visages)
        color = (0, 255, 0) if num_faces == 1 else (0, 0, 255)
        cv.rectangle(img, start_point, end_point, color, 3)

    # Afficher l'image finale
    cv.imshow('TrustGrade Detection', img)
    cv.waitKey(0)
    cv.destroyAllWindows()