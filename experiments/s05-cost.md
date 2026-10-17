# Palier 5 : coût d'un modèle (paramètres, FLOPs, latence)

Rapport du 15 octobre 2026. Code : `steps/s05_cost.py`.

## 1. Question

Comment se comparent les métriques de coût (nombre de paramètres, FLOPs/MACs théoriques, latence d'inférence réelle sur mps et CPU pour batch 1 et batch 64) à travers une grille de 6 architectures CNN sur CIFAR-10, et dans quelle mesure le classement théorique par FLOPs concorde-t-il avec la latence mesurée ?

## 2. Prédiction (écrite par Jean avant toute mesure, jamais modifiée après)

On sait avec l'etape precedente faire x2 wide fait x4 parametres, augmenter deep x2 fait environ x3 parametres mais que les macs sont a peu pres equivalents. La latence n'avait pas ete estimés a cause d'une seul seed mais ce qui en etait sortie est qu'augmenter en wide influence moins le temps de calcul que d'augmenter en deep. Mais deep seul semblait plus prometteur en precision que wide seul. L'interet va etre de voir si la liaison des deux est propice a de meilleurs perfs, je pense qu'un x2 wide et x2 deep va augmenter l'accuracy d'encore 5% environ par rapport au deep precedent. Je pense que FLOPS ne va pas tout dire sur la latence. Ce ne sera pas suffisant a savoir la latence de chaques modele meme si ca va donner une ordre de grandeur. Je pense que ce sont vraiemnt les tests qui peuvent donner la latence.

## 3. Dispositif

* Grille de 6 configurations :
  * `tiny` : 2 étages (canaux 16, 32), noyaux 3x3, pool.
  * `s03` : 3 étages (canaux 32, 64, 128), noyaux 3x3, pool.
  * `wide` : 3 étages (canaux 64, 128, 256), noyaux 3x3, pool.
  * `deep` : 3 étages (canaux 32, 64, 128), 2 blocs par étage, noyaux 3x3, pool en fin d'étage.
  * `kernel5` : 3 étages (canaux 32, 64, 128), noyaux 5x5, pool.
  * `late_pool` : 3 étages (canaux 32, 64, 128), pool retardé au 3e étage (`pool=False` sur l'étage 2).
* Paramètres d'entraînement :
  * Époques : 4
  * Optimiseur : `AdamW(lr=3e-3, weight_decay=5e-4)`, scheduler `OneCycleLR`
  * Taille de lot d'entraînement : 128
* Mesure du coût :
  * Paramètres : `sum(p.numel() for p in model.parameters())`
  * FLOPs / MACs : `FlopCounterMode` de PyTorch (1 MAC = 2 FLOPs, batch 1)
  * Latence : boucle manuelle avec warm-up (10 passes), `synchronize(device)` et médiane sur 50 passes, batchs 1 et 64, sur `mps` et `cpu`.
* Graines : 3 graines pour les résultats retenus (graines 0, 1, 2 avec `--test`), plus 1 graine d'exploration préalable (graine 0).
* Device et versions (Python, torch) : mps (Mac M4 16 Go), Python 3.12.13, PyTorch 2.14.1.
* Temps d'entraînement : ~163 s pour l'exploration, ~546 s pour les 3 graines finales (~9 min).

## 4. Résultats

Chiffres finaux retenus sur 3 graines (graines 0, 1, 2 ; 4 époques ; moyenne et écart-type sur le jeu de validation de 5 000 images et le jeu test de 10 000 images de CIFAR-10 ; médiane des latences sur les 3 runs) :

| Configuration | Paramètres | MACs (M) | FLOPs (M) | val_acc (3 graines) | test_acc (3 graines) | mps b1 (ms) | mps b64 (ms) | cpu b1 (ms) | cpu b64 (ms) | Temps train moy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tiny` | 5 466 | 1,62 | 3,24 | 45,23 % ± 0,77 % | 44,75 % ± 0,93 % | 0,680 ms | 2,456 ms | 0,142 ms | 4,243 ms | 11,2 s |
| `s03` | 94 762 | 10,32 | 20,65 | 67,69 % ± 0,16 % | 67,43 % ± 0,13 % | 0,419 ms | 2,725 ms | 0,258 ms | 10,383 ms | 19,7 s |
| `late_pool` | 94 762 | 24,48 | 48,96 | 64,43 % ± 0,20 % | 64,40 % ± 0,41 % | 0,466 ms | 4,396 ms | 0,346 ms | 17,644 ms | 32,0 s |
| `kernel5` | 260 138 | 28,67 | 57,35 | 72,16 % ± 0,10 % | 71,95 % ± 0,20 % | 0,642 ms | 4,882 ms | 0,377 ms | 14,451 ms | 33,5 s |
| `deep` | 288 746 | 38,63 | 77,27 | 81,85 % ± 0,34 % | 80,95 % ± 0,23 % | 0,866 ms | 5,238 ms | 0,530 ms | 33,928 ms | 40,9 s |
| `wide` | 373 834 | 39,52 | 79,04 | 72,05 % ± 0,17 % | 71,79 % ± 0,55 % | 0,690 ms | 5,957 ms | 0,325 ms | 26,032 ms | 45,1 s |

Recoupement ponctuel sur CPU (batch 1, architecture `s03`) :
* `torch.utils.benchmark.Timer` (CPU, batch 1) : médiane 0,234 ms
* Boucle manuelle avec `time.perf_counter()` : 0,227 ms (écart inférieur à 3 %)

## 5. Surprises et échecs

* Le CPU bat systématiquement le GPU (mps) sur batch 1 : pour chaque architecture, la latence CPU en batch 1 est 1,4 à 4,8 fois plus faible que sur mps (par exemple 0,142 ms contre 0,680 ms pour `tiny`, mais 0,346 ms contre 0,466 ms pour `late_pool`). Sur un petit volume de calcul (une seule image 32x32), le temps d'inférence GPU est dominé par l'overhead de soumission (pont Python vers Metal et ordonnancement GPU), alors que le CPU traite les données immédiatement en cache L1/L2.
* À l'inverse, sur batch 64, le GPU sature ses unités de calcul et écrase le CPU : mps est 1,7 à 6,5 fois plus rapide que le CPU (par exemple 5,238 ms contre 33,928 ms pour `deep`, mais 2,456 ms contre 4,243 ms pour `tiny`).
* Divergence entre FLOPs et latence selon le matériel :
  * Sur `mps` en batch 64, `deep` (77,27 M FLOPs) est plus rapide que `wide` (79,04 M FLOPs) : 5,238 ms contre 5,957 ms.
  * Mais sur `cpu` en batch 64, c'est l'inverse : `wide` est plus rapide que `deep` (26,032 ms contre 33,928 ms), car les 6 convolutions séquentielles de `deep` créent plus de transferts mémoire et de défauts de cache CPU que les convolutions plus larges et vectorisables de `wide`. Le classement par FLOPs ne prédit donc pas fidèlement la latence.
* `late_pool` : même nombre de paramètres que `s03` (94 762), mais 2,37 fois plus de FLOPs (48,96 M contre 20,65 M) parce que la carte intermédiaire reste en 16x16 au lieu de 8x8. Pourtant, sa précision après 4 époques est inférieure à celle de `s03` (64,43 % contre 67,69 % en val ; 64,40 % contre 67,43 % en test), démontrant que gonfler les FLOPs sans augmenter la capacité paramétrique ne garantit aucun gain.
* Efficacité spectaculaire de `deep` : avec 288k paramètres et 77 M FLOPs, `deep` domine très nettement toutes les autres configurations en précision (80,95 % sur le jeu test, soit +9,16 points d'avance sur `wide` à 71,79 % et sur `kernel5` à 71,95 %), tout en ayant moins de paramètres et une latence GPU inférieure à `wide`. L'intuition de Jean sur la supériorité de la profondeur est confirmée de manière frappante.

## 6. Questions à savoir traiter sans le dépôt

1. Quelle est la différence entre MACs et FLOPs, et que compte (et ne compte pas) `FlopCounterMode` ?
2. Pourquoi faut-il appeler `torch.mps.synchronize()` avant d'arrêter le chronomètre sur mps, et que mesurerait-on sans cela ?
3. Pourquoi le classement par FLOPs et le classement par latence peuvent-ils diverger, et lequel des deux utiliserais-tu comme objectif dans un NAS ciblant un matériel précis ?
