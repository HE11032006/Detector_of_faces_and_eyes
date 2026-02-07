import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def detect_eyes(IMAGE_PATH, MODEL_PATH = 'face_landmarker.task', display = True):

    # Configuration du FaceLandmarker pour détecter les points du visage
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # Charger l'image
        img = cv.imread(IMAGE_PATH)
        if img is None:
            raise IOError(f"Impossible de charger l'image {IMAGE_PATH}")

        # Convertir BGR → RGB pour MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv.cvtColor(img, cv.COLOR_BGR2RGB)
        )

        # Détecter les landmarks du visage
        result = landmarker.detect(mp_image)

        # Indices des landmarks pour les yeux (MediaPipe Face Mesh)
        # Œil gauche :
        LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
        # Œil droit :
        RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

        num_faces = len(result.face_landmarks)

        # Validation du nombre de visages
        if num_faces > 1:
            print(f"ALERTE : {num_faces} visages détectés !")
        elif num_faces == 0:
            print("ALERTE : Aucun visage visible !")
        else:
            print("Un seul visage détecté. Tout est OK.")

        # Dessiner les yeux pour chaque visage détecté
        eyes_landmarks = []
        for face_landmarks in result.face_landmarks:
            h, w = img.shape[:2]

            left_eye_points = []
            right_eye_points = []

            # Dessiner l'œil gauche en vert
            for idx in LEFT_EYE:
                landmark = face_landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv.circle(img, (x, y), 2, (0, 255, 0), -1)
                left_eye_points.append((x, y))

            # Dessiner l'œil droit en bleu
            for idx in RIGHT_EYE:
                landmark = face_landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv.circle(img, (x, y), 2, (255, 0, 0), -1)
                right_eye_points.append((x, y))

            eyes_landmarks.append({
                'left_eye': left_eye_points,
                'right_eye': right_eye_points
            })

            print(f"Yeux détectés pour le visage : œil gauche ({len(LEFT_EYE)} points), œil droit ({len(RIGHT_EYE)} points)")

        # Afficher l'image avec les yeux marqués
        if display:
            cv.imshow('Eye Detection', img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return img, num_faces, eyes_landmarks


