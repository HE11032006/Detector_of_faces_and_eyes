# Detecteur_de_visage

Ce projet est un outil simple pour détecter des visages et des yeux sur des images en utilisant MediaPipe et OpenCV. Il regroupe des détecteurs préexistants (détection de visage et détection de landmarks pour les yeux) et fournit une petite interface en ligne de commande pour exécuter ces détections sur une ou plusieurs images.

Principes du projet
-------------------
- Le projet utilise deux composants principaux :
  - un détecteur de visages (modèle TensorFlow Lite, ex. `detector.tflite`),
  - un face landmarker (modèle MediaPipe, ex. `face_landmarker.task`) pour extraire les repères du visage (landmarks) et détecter les yeux.
- Les fonctions de détection sont modulaires : `detect_faces()` et `detect_eyes()` retournent les résultats (bboxes et landmarks) sans afficher de fenêtre si on le souhaite. Un script assembleur (`main.py`) combine ces résultats et produit une image annotée unique.

Exécution (prérequis)
---------------------
Supposons que vous avez :
- créé et activé un environnement virtuel Python (recommandé),
- installé les dépendances listées dans `requirements.txt`,
- placé les modèles requis dans le répertoire du projet.

Exemples de commandes (exécuter dans le répertoire du projet) :

```bash
# activer un environnement virtuel (exemple)
python -m venv .venv
source .venv/bin/activate

# installer les dépendances
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# afficher l'aide de l'outil (toutes les options)
python main.py --help
```

Modèles requis
--------------
- `detector.tflite` : modèle TFLite pour la détection de visages. Si vous ne l'avez pas, fournissez-en un compatible ou placez `detector.tflite` à la racine du projet.
- `face_landmarker.task` : modèle MediaPipe pour les landmarks du visage. Téléchargez-le depuis le dépôt officiel MediaPipe Models et placez-le à la racine du projet.

Si un de ces fichiers manque ou est vide, la détection échouera (erreur de lecture / fichier introuvable).

Utilisation de l'interface CLI
-----------------------------
Le script principal `main.py` fournit les options suivantes :

- `-f` / `--faces`  : détecter uniquement les visages
- `-e` / `--eyes`   : détecter uniquement les yeux
- `-b` / `--both`   : détecter visages ET yeux simultanément
- `--no-display`    : ne pas afficher les fenêtres (utile pour traitement en batch)
- `--save`          : sauvegarder l'image annotée dans `--output-dir`
- `--output-dir`    : répertoire où sauvegarder les images annotées (par défaut `output`)

Exemples :

```bash
# détecter les visages sur une image et afficher le résultat
python main.py --faces One_face0.jpg

# détecter les yeux sur plusieurs images (glob peut être utilisé par le shell)
python main.py --eyes img1.jpg img2.jpg

# détecter visages et yeux, ne pas afficher et sauvegarder les résultats
python main.py --both image.jpg --no-display --save --output-dir results
```

Traitement de plusieurs images
------------------------------
Vous pouvez passer plusieurs chemins d'images en arguments ; le script les traite séquentiellement, et, si demandé, sauvegarde chaque image annotée avec un suffixe indiquant le mode de détection.

Score de confiance et seuils
---------------------------
- Chaque détection produite par les modèles inclut un score de confiance (valeur entre 0 et 1).
- Par défaut le code utilise un seuil permissif (0.5) pour ne pas rater de vraies détections. Si vous voulez être strict, augmentez le seuil (par exemple 0.95) pour réduire les faux positifs.
- Attention aux images floues ou aux motifs :
  - Sur des images très floues, des objets non humains (par exemple une voiture) peuvent parfois être détectés comme visage si le modèle confond des textures. Augmenter le seuil de confiance réduit ce risque.
  - Les modèles modernes restent toutefois sensibles aux conditions d'éclairage, aux poses de profil et à la qualité de la photo.

Images à analyser
-----------------
- Placez vos images directement dans le dossier du projet ou fournissez des chemins absolus/relatifs à `main.py`.
- Pour un traitement organisé, créez un dossier `images/` et passez ses fichiers au script :

```bash
python main.py --both images/*.jpg --save
```

Bonnes pratiques
----------------
- Activez et utilisez un environnement virtuel pour éviter les conflits de dépendances.
- Vérifiez que `face_landmarker.task` n'est pas vide après téléchargement (taille attendue ~ plusieurs Mo). Un fichier vide provoquera une erreur.
- Pour l'analyse automatique (sans affichage), utilisez `--no-display` et `--save` pour récupérer les images annotées.

Support et développement
-----------------------
- Le code est modulaire : les fonctions `detect_faces()` et `detect_eyes()` peuvent être réutilisées ou étendues (par exemple pour détecter iris, sourcils ou mesurer l'ouverture des yeux).
- Si vous rencontrez des faux positifs fréquents, essayez d'augmenter le seuil de confiance, d'améliorer la qualité des images ou d'utiliser un modèle DNN différent.

---

Version actuelle : code fourni dans ce dépôt. Pour toute question ou bug, modifiez les scripts ou ouvrez une issue dans votre gestionnaire de code.

---

Auteur
------

Nom : Michel HOUESSOU

Email : michelhss11@gmail.com
