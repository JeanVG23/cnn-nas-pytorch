# Palier 3 : CNN sur CIFAR-10

Rapport du 13 octobre 2026. Code : `steps/s03_cnn_cifar.py`.

## 1. Question

Quelle précision de validation et de test un petit CNN (3 blocs Conv2d-BatchNorm2d-ReLU-MaxPool2d de 32, 64 et 128 canaux, suivi d'un pooling global adaptatif et d'une couche linéaire) atteint-il sur CIFAR-10 après 15 époques avec AdamW et OneCycleLR (lr=3e-3, batch_size=128), et combien de paramètres compte ce réseau ?

## 2. Prédiction (écrite par Jean avant toute mesure, jamais modifiée après)

On devrait arriver a 78% de precision, le reseau devrait compter 60k parametres.

## 3. Dispositif

* Paramètres : 3 blocs `Conv2d(k=3, pad=1, bias=False)` - `BatchNorm2d` - `ReLU` - `MaxPool2d(2)` (canaux 32, 64, 128), tête `AdaptiveAvgPool2d((1, 1))` - `Flatten` - `Linear(128, 10)`. Optimiseur `AdamW(lr=3e-3, weight_decay=5e-4)`, scheduler `OneCycleLR`, 15 époques, batch_size=128. Augmentation à l'entraînement (`RandomCrop(32, padding=4)`, `RandomHorizontalFlip`).
* Graines : 0 (exploration sans `--test`), puis 0, 1, 2 (chiffres retenus avec `--test`).
* Device et versions : mps (Mac M4 16 Go), Python 3.12.13, PyTorch 2.14.1.
* Nombre de paramètres : 94 762 (94 752 poids et 10 biais).
* Temps d'entraînement : 101,7 s en moyenne par graine pour les 3 runs finaux (total ~305 s sur 3 graines).

## 4. Résultats

* Exploration (1 graine, graine 0, sans `--test`) : `val_loss = 0.6289`, `val_acc = 78.18 %`.
* Synthèse sur 3 graines (runs finaux avec `--test`) :
  * Graine 0 : `val_acc = 78.18 %`, `val_loss = 0.6289` ; `test_acc = 77.69 %`, `test_loss = 0.6541`
  * Graine 1 : `val_acc = 77.88 %`, `val_loss = 0.6187` ; `test_acc = 77.53 %`, `test_loss = 0.6478`
  * Graine 2 : `val_acc = 78.52 %`, `val_loss = 0.6223` ; `test_acc = 77.59 %`, `test_loss = 0.6480`
  * Moyenne retenue validation (3 graines, 5 000 images) : `val_acc = 78.19 %` (dispersion minime de +-0.32 point), `val_loss = 0.6233`
  * Moyenne retenue test (3 graines, 10 000 images) : `test_acc = 77.60 %` (dispersion infime de +-0.08 point), `test_loss = 0.6500`
  * Sorties après sauvegarde et rechargement de checkpoint (`data/checkpoints/s03_cnn_cifar.pt`) : strictement identiques (`torch.allclose == True`).

## 5. Surprises et échecs

* Précision : La prédiction de Jean (78 %) est remarquablement exacte face aux mesures réelles (78,19 % en validation et 77,60 % en test). L'écart validation-test n'est que de 0,59 point, démontrant une très bonne généralisation sans surapprentissage grâce aux augmentations et au pooling global.
* Paramètres : Jean estimait 60k paramètres contre 94 762 en réalité. L'écart vient de la croissance quadratique du nombre de poids avec les canaux : le troisième bloc convolutif concentre à lui seul 73 728 poids (soit 77,8 % des paramètres totaux du modèle).
* Incident d'exécution : Au premier lancement, l'extraction de l'archive `cifar-10-python.tar.gz` a échoué avec `OSError: [Errno 28] No space left on device` car le disque du Mac n'avait plus que 1,1 Go disponible. Après libération d'espace disque par Jean (15 Go libres) et nettoyage du dossier partiellement extrait, l'entraînement s'est déroulé sans encombre.

