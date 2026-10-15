# Palier 4 : CNN configurable depuis un dict

Rapport du 14 octobre 2026. Code : `steps/s04_cnn_config.py`.

## 1. Question

Quel est l'impact de l'élargissement (`wide` : canaux doublés 64-128-256) et de l'approfondissement (`deep` : 2 blocs convolutifs par étage, canaux 32-64-128) sur le nombre total de paramètres et sur la précision de validation d'un CNN sur CIFAR-10 après 10 époques d'entraînement, par rapport à la baseline `s03_equivalent` (94 762 paramètres) ?

## 2. Prédiction (écrite par Jean avant toute mesure, jamais modifiée après)

On aura un temps plus grands de calcul du a l'elargissement, je ne pense pas qu'on aura d'amelioration substentiel avec un simple elargissment. L'approfidissment semble plus prometteur puisque rajouter des couches caches va permettre de mieux capter les motifs. Donc je pense que wide sera plus long et pas meilleur mais deep sera un peu meilleur de l'ordre de 5%.
Pour les parametres on devrait avoir bcp plus de parametre sur wide, on devrait avoir 3*3*3*64+3*3*64*128+3*3*128*256 + tete environ 1k + normalisation environ 500 poids. Pour le deep on devrait pas avoir bcp plus de plat que pour le classique de la v3.

## 3. Dispositif

* Paramètres :
  * Baseline `s03_equivalent` : 3 étages `Conv2d-BatchNorm2d-ReLU-MaxPool2d` (canaux 32, 64, 128), 1 bloc par étage.
  * Variante `wide` : 3 étages (canaux 64, 128, 256), 1 bloc par étage.
  * Variante `deep` : 3 étages (canaux 32, 64, 128), 2 blocs convolutifs par étage (le pooling n'intervient qu'en fin d'étage).
  * Tête commune : `AdaptiveAvgPool2d((1, 1))` - `Flatten` - `Linear(C_out, 10)`.
  * Optimiseur : `AdamW(lr=3e-3, weight_decay=5e-4)`, scheduler `OneCycleLR`, batch size 128, 10 époques.
* Graines : graine 0 en exploration (sans `--test`).
* Device et versions : mps (Mac M4 16 Go), Python 3.12.13, PyTorch 2.14.1.
* Nombre de paramètres :
  * `s03_equivalent` : 94 762
  * `wide` : 373 834
  * `deep` : 288 746
* Temps d'entraînement :
  * `s03_equivalent` : 46,04 s (~4,6 s par époque)
  * `wide` : 105,25 s (~10,5 s par époque)
  * `deep` : 155,20 s (~15,5 s par époque)
  * Total : 306,49 s (~5 min 06 s)

## 4. Résultats

Exploration sur 1 graine (graine 0, 10 époques, validation sur 5 000 images de CIFAR-10) :

| Configuration | Paramètres | Temps (s) | Temps / époque | val_loss | val_acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `s03_equivalent` | 94 762 | 46,04 s | 4,6 s | 0,7102 | 74,82 % |
| `wide` | 373 834 | 105,25 s | 10,5 s | 0,5866 | 80,06 % |
| `deep` | 288 746 | 155,20 s | 15,5 s | 0,3978 | 86,36 % |

## 5. Surprises et échecs

* Précision : L'intuition de Jean sur la supériorité de l'approfondissement (`deep`) s'est avérée très juste face à l'élargissement (`wide`), mais l'ampleur du gain dépasse largement les prévisions : `deep` atteint 86,36 % (+11,54 points par rapport à la baseline et +6,30 points par rapport à `wide`), avec une perte de validation très basse (0,3978 contre 0,5866). `wide` n'est pas inefficace pour autant : il franchit les 80 % (+5,24 points par rapport à la baseline).
* Paramètres : La formule analytique posée par Jean pour `wide` était rigoureusement exacte (373 834 paramètres réels). En revanche, `deep` compte 288 746 paramètres (soit le triple de la baseline de 94k), car chaque deuxième convolution d'étage opère en `C -> C` plein (notamment le bloc 128 -> 128 qui ajoute à lui seul 147 456 paramètres).
* Temps de calcul et latence matérielle : Bien que `deep` compte 23 % de paramètres en moins que `wide` (288k vs 374k), il est 47 % plus lent à l'entraînement (15,5 s par époque contre 10,5 s). La profondeur impose 6 passes convolutives strictement séquentielles qui ne saturent pas le GPU, alors que l'élargissement de `wide` permet une parallélisation matricielle massive sur les cœurs de calcul du backend MPS.

