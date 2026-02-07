import cv2 as cv
from detect_faces import detect_faces
from detect_eyes import detect_eyes


def detect_face_and_eyes(image_path):

    print("=== Détection des visages ===")
    img_faces, num_faces, bboxes = detect_faces(image_path, display=False)

    print("\n=== Détection des yeux ===")
    img_eyes, num_faces_eyes, eyes_landmarks = detect_eyes(image_path, display=False)

    # Combiner les deux détections sur une même image
    img_combined = cv.imread(image_path)

    # Dessiner les visages (rectangles verts/rouges)
    for bbox in bboxes:
        color = (0, 255, 0)
        cv.rectangle(img_combined,
                     (bbox['x'], bbox['y']),
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                     color, 3)

    # Dessiner les yeux (points verts et bleus)
    for eyes in eyes_landmarks:
        for x, y in eyes['left_eye']:
            cv.circle(img_combined, (x, y), 2, (0, 255, 0), -1)
        for x, y in eyes['right_eye']:
            cv.circle(img_combined, (x, y), 2, (0, 255, 0), -1)

    # Afficher le résultat final
    cv.imshow('Face & Eyes Detection', img_combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img_combined, num_faces, bboxes, eyes_landmarks


if __name__ == "__main__":
    detect_face_and_eyes('Two_face0.jpg')




