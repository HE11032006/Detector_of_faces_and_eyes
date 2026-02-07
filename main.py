"""
Comment l'utiliser ?
Utilisation:
    python main.py --faces image.jpg                    # Détection visages uniquement
    python main.py --eyes image.jpg                     # Détection yeux uniquement
    python main.py --both image.jpg                     # Détection visages + yeux
    python main.py --faces img1.jpg img2.jpg img3.jpg   # Plusieurs images
    python main.py --both *.jpg                         # Toutes les images JPG
"""

import argparse
import sys
import os
from detect_faces import detect_faces
from detect_eyes import detect_eyes
from detect_faces_and_eyes import detect_face_and_eyes


def main():
    parser = argparse.ArgumentParser(
        description="Détecteur simplement de visages et yeux.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --faces One_face0.jpg
  %(prog)s --eyes Two_face0.jpg
  %(prog)s --both One_face0.jpg
  %(prog)s --faces img1.jpg img2.jpg img3.jpg
  %(prog)s -f *.jpg
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-f', '--faces',
        action='store_true',
        help='Détecter uniquement les visages'
    )
    group.add_argument(
        '-e', '--eyes',
        action='store_true',
        help='Détecter uniquement les yeux'
    )
    group.add_argument(
        '-b', '--both',
        action='store_true',
        help='Détecter visages ET yeux simultanément'
    )

    # Arguments pour les images
    parser.add_argument(
        'images',
        nargs='+',
        help='Chemin(s) vers une ou plusieurs images'
    )

    # Options supplémentaires
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Ne pas afficher les images (sauvegarder seulement)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Sauvegarder les images annotées'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Répertoire de sortie pour les images sauvegardées (défaut: output)'
    )

    args = parser.parse_args()

    # Créer le répertoire de sortie si nécessaire
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    # Traiter chaque image
    total_images = len(args.images)
    print(f"=== Traitement de {total_images} image(s) ===\n")

    for idx, image_path in enumerate(args.images, 1):
        print(f"[{idx}/{total_images}] Traitement de : {image_path}")

        # Vérifier que l'image existe
        if not os.path.exists(image_path):
            print(f"ERREUR : Fichier introuvable : {image_path}")
            continue

        try:
            display = not args.no_display

            # Choix du type de détection
            if args.faces:
                print("  -- Mode : Détection de visages")
                img, num_faces, bboxes = detect_faces(image_path, display=display)
                result_type = "faces"

            elif args.eyes:
                print("  -- Mode : Détection des yeux")
                img, num_faces, eyes_landmarks = detect_eyes(image_path, display=display)
                result_type = "eyes"

            elif args.both:
                print("  -- Mode : Détection visages + yeux")
                img, num_faces, bboxes, eyes_landmarks = detect_face_and_eyes(image_path)
                result_type = "both"

            # Sauvegarder si demandé
            if args.save:
                import cv2 as cv
                # Générer nom de fichier de sortie
                basename = os.path.basename(image_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(args.output_dir, f"{name}_{result_type}{ext}")
                cv.imwrite(output_path, img)
                print(f"   Sauvegardé : {output_path}")

            print(f"  Visages détectés : {num_faces}\n")

        except Exception as e:
            print(f"  ERREUR : {str(e)}\n")
            continue

    print(f"=== Traitement terminé : {total_images} image(s) ===")


if __name__ == "__main__":
    main()

