# Palier 6 : Random Search et Frontière de Pareto

Rapport du 16 octobre 2026. Code : `steps/s06_pareto.py`.

## 1. Question

Dans quelle mesure une recherche aléatoire (Random Search de 15 architectures) permet-elle d'isoler des architectures non-dominées sur CIFAR-10, et observe-t-on une divergence entre le front de Pareto théorique (Précision vs FLOPs) et le front de Pareto réel (Précision vs Latence matérielle) ?

## 2. Prédiction (écrite par Jean avant toute mesure, jamais modifiée après)

Sur des petit batch on verra des grandes diffs etnre pareto theorique et pareto reel parce que comme vu precedemment, sur petit batch on ne peut pas se fier aux flops pour determiner la latence et donc on aura une divergence entre les deux. Le random search est un algorithme assez bete, donc il y aura beaucoup de tests d'architecture dominée sachant qu'on a deja l'infomration qu'elle sera probableemnt dominée, c'est pour ca que je pense que si on veut optimiser il faudrati utiliser peut etre une BO je ne sais pas si ca existe en NN. On aura quand meme des modeles frontier prometteurs.

## 3. Dispositif

* Espace de recherche (Search Space) :
  * 2 à 4 étages convolutifs.
  * Canaux par étage choisis dans {16, 32, 64, 128} sous contrainte pyramidale croissante.
  * Profondeur par étage : 1 ou 2 blocs convolutifs.
  * Noyaux : 3x3 ou 5x5.
  * Sous-échantillonnage (pooling) : True ou False, garantissant une taille spatiale terminale >= 2x2.
  * Tête commune : `AdaptiveAvgPool2d((1, 1))` - `Flatten` - `Linear(C_out, 10)`.
* Budget et échantillonnage :
  * 15 architectures distinctes échantillonnées avec graine fixe (`seed=42`).
  * Entraînement proxy : 3 époques par architecture sur CIFAR-10, optimiseur AdamW (lr=3e-3, weight_decay=5e-4), scheduler OneCycleLR, batch size 128.
* Métriques multi-objectifs évaluées :
  * Performance : `val_acc` sur les 5 000 images de validation de CIFAR-10.
  * Coût théorique : nombre de paramètres et FLOPs/MACs pour une image (FlopCounterMode, batch 1).
  * Coût matériel réel : latence médiane en millisecondes sur `mps` (batch 64 et batch 1) et sur `cpu` (batch 1 et batch 64).
* Device et versions : mps (Mac M4 16 Go), Python 3.12.13, PyTorch 2.14.1.
* Temps total d'entraînement de la grille : 2 683,9 s (~44 min 44 s pour les 15 modèles).

## 4. Résultats

Résultats complets des 15 architectures tirées au sort, triées par précision de validation (`val_acc`) décroissante :

| ID | Architecture | Paramètres | MACs (M) | FLOPs (M) | val_acc | mps b64 (ms) | mps b1 (ms) | cpu b64 (ms) | cpu b1 (ms) | Temps train |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `arch_14` | `2x128k3P > 1x128k5P > 2x128k5- > 1x128k5P` | 1 792 138 | 338,04 | 676,07 | **82,36 %** | 22,80 ms | 1,22 ms | 122,28 ms | 1,36 ms | 171,4 s |
| `arch_09` | `1x32k5P > 2x128k5P > 2x128k5P > 1x128k3P` | 1 483 690 | 188,32 | 376,64 | **79,36 %** | 13,63 ms | 1,63 ms | 64,43 ms | 1,17 ms | 99,4 s |
| `arch_02` | `2x128k5P > 2x128k5- > 2x128k3P > 1x128k5P` | 1 945 994 | 740,69 | 1 481,38 | 79,30 % | 53,58 ms | 1,93 ms | 178,92 ms | 2,50 ms | 315,4 s |
| `arch_06` | `2x64k5- > 1x64k5P > 2x128k5- > 2x128k3P` | 1 121 610 | 447,42 | 894,83 | 79,20 % | 30,67 ms | 1,03 ms | 135,90 ms | 1,40 ms | 237,7 s |
| `arch_04` | `1x64k3P > 2x128k5- > 2x128k5P > 1x128k3P` | 1 585 482 | 378,21 | 756,42 | 76,86 % | 25,68 ms | 1,17 ms | 129,51 ms | 1,21 ms | 191,9 s |
| `arch_05` | `2x32k5- > 2x128k3P > 2x128k3- > 1x128k3P` | 657 386 | 330,66 | 661,33 | 76,76 % | 21,50 ms | 1,11 ms | 136,39 ms | 1,01 ms | 174,2 s |
| `arch_03` | `2x128k3P > 2x128k3- > 1x128k3P` | 595 850 | 267,78 | 535,56 | 76,18 % | 19,41 ms | 0,85 ms | 114,52 ms | 0,98 ms | 141,2 s |
| `arch_08` | `1x128k3- > 2x128k3- > 1x128k5- > 2x128k3P` | 1 005 706 | 1 026,95 | 2 053,90 | 73,54 % | 61,96 ms | 1,35 ms | 272,41 ms | 2,70 ms | 463,8 s |
| `arch_15` | `2x16k5P > 1x32k3- > 1x32k5- > 1x64k3P` | 57 210 | 20,23 | 40,47 | **66,84 %** | 3,97 ms | 1,19 ms | 13,11 ms | 0,36 ms | 20,0 s |
| `arch_10` | `1x128k3- > 2x128k5P` | 824 714 | 842,40 | 1 684,80 | 65,24 % | 60,49 ms | 1,35 ms | 166,86 ms | 2,42 ms | 398,3 s |
| `arch_01` | `1x16k5P > 1x32k3- > 1x32k3P > 1x32k3P` | 24 794 | 5,36 | 10,72 | **63,66 %** | 1,96 ms | 1,07 ms | 6,06 ms | 0,25 ms | 10,2 s |
| `arch_11` | `2x128k5- > 1x128k5P` | 830 858 | 848,69 | 1 697,38 | 63,20 % | 60,93 ms | 1,37 ms | 166,22 ms | 2,36 ms | 405,9 s |
| `arch_13` | `1x32k3- > 2x32k3P` | 19 818 | 19,76 | 39,52 | 53,78 % | 3,21 ms | 0,79 ms | 20,82 ms | 0,23 ms | 21,3 s |
| `arch_07` | `1x16k3- > 2x16k5P` | 13 498 | 13,55 | 27,10 | 51,20 % | 2,91 ms | 1,05 ms | 11,31 ms | 0,28 ms | 14,8 s |
| `arch_12` | `1x32k5P > 1x128k3P` | 40 874 | 11,90 | 23,79 | 50,66 % | 4,42 ms | 0,87 ms | 12,93 ms | 0,55 ms | 18,4 s |

### Frontières de Pareto identifiées

1. **Frontière Précision vs FLOPs théoriques** : `arch_01`, `arch_15`, `arch_09`, `arch_14` (4 modèles non-dominés sur 15).
2. **Frontière Précision vs Latence mps batch 64** : `arch_01`, `arch_15`, `arch_09`, `arch_14` (identique au front FLOPs : à fort parallélisme, la puissance de calcul brute dicte la latence).
3. **Frontière Précision vs Latence mps batch 1 (inférence temps réel)** : `arch_13`, `arch_03`, `arch_06`, `arch_14`.
   BASCULEMENT MAJEUR : 3 des 4 architectures du front théorique (`arch_01`, `arch_15`, `arch_09`) sont éliminées et dominées en latence batch 1 !
4. **Frontière Précision vs Latence CPU batch 1** : `arch_13`, `arch_01`, `arch_15`, `arch_03`, `arch_05`, `arch_09`, `arch_14`.

## 5. Surprises et enseignements

* Confirmation éclatante de la divergence FLOPs vs Latence sur petit lot :
  Comme prédit par Jean, les FLOPs ne prédisent pas la latence en batch 1.
  Par exemple, `arch_01` compte seulement 5,36 M MACs mais met 1,066 ms à s'exécuter sur mps en batch 1. À l'opposé, `arch_13` compte 19,76 M MACs (presque 4 fois plus !) mais s'exécute en seulement 0,785 ms sur mps. L'overhead d'ordonnancement de 4 petits étages pénalise plus `arch_01` que le volume de calcul de `arch_13`.
  Conséquence en ingénierie : un ingénieur NAS qui optimiserait uniquement sur les FLOPs déploierait une architecture sous-optimale pour un capteur temps réel.
* Inefficience massive du Random Search et justification de l'Optimisation Bayésienne (BO) :
  Sur 15 tirages, 11 modèles (73 %) sont strictement dominés. Les trois modèles géants sans pooling initial (`arch_08`, `arch_10`, `arch_11`) ont consommé à eux seuls 1 268 secondes de calcul (soit 47 % du budget de calcul total du palier !) pour des précisions médiocres comprises entre 63 % et 73 %.
  Comme pressenti par Jean, un algorithme guidé par modèle de substitution (comme BOHB ou BANANAS en NAS bayésien) aurait appris dès le premier échec qu'omettre le pooling sur 128 canaux est une impasse, évitant de gaspiller du temps de GPU sur `arch_10` et `arch_11`.
* Victoire de l'architecture `arch_14` :
  Avec 1,79M paramètres et 338 M MACs, elle franchit les 82,36 % en seulement 3 époques d'entraînement. Sa conception (2 étages avec pooling précoce, puis un étage de 128 canaux sans pooling pour accumuler de la profondeur sémantique) en fait le modèle dominant sur l'ensemble des métriques de précision.
* Efficacité spectaculaire de `arch_09` :
  1,48M paramètres, 188 M MACs, 79,36 % de précision en seulement 99 secondes d'entraînement. Elle domine strictement `arch_02` (79,30 %, 740 M MACs, 315 s d'entraînement), prouvant qu'avoir 4 fois moins de calculs permet une convergence plus rapide et une meilleure précision.

## 6. Questions à savoir traiter sans le dépôt

1. Définis formellement la dominance de Pareto entre deux architectures A et B pour un problème multi-objectifs (maximiser la précision, minimiser la latence).
2. Pourquoi le Random Search constitue-t-il une ligne de base (baseline) redoutable en NAS face à des algorithmes plus complexes (RL, algorithmes génétiques, gradients continus DARTS) ?
3. Pourquoi optimiser un modèle uniquement sur les FLOPs théoriques peut-il conduire à déployer une architecture sous-optimale en latence sur une cible matérielle donnée (ex: GPU embarqué ou microcontrôleur) ?
