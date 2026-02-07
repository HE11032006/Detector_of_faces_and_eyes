import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def detect_faces(IMAGE_PATH, MODEL_PATH = 'detector.tflite', display = True):
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
        bboxes = []
        i = 0
        for detection in result.detections:
            bbox = detection.bounding_box
            start_point = int(bbox.origin_x), int(bbox.origin_y)
            end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)

            # Dessiner le rectangle (Vert si OK, Rouge si plusieurs visages)
            color = (0, 255, 0) if num_faces == 1 else (0, 0, 255)
            cv.rectangle(img, start_point, end_point, color, 3)

            # Ajouter bbox à la liste
            bboxes.append({
                'x': int(bbox.origin_x),
                'y': int(bbox.origin_y),
                'width': int(bbox.width),
                'height': int(bbox.height)
            })

            # extraire les visages de l'image principale
            face = img[int(bbox.origin_y):int(bbox.origin_y + bbox.height), int(bbox.origin_x):int(bbox.origin_x + bbox.width)]

            if display:
                cv.imshow(f'Face{i}', face)
            i += 1

        if display:
            cv.imshow('TrustGrade Detection', img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return img, num_faces, bboxes
