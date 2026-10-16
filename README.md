# cnn_pytorch : un mini-NAS PyTorch construit pas à pas

Un parcours en 5 paliers pour devenir opérationnel sur PyTorch : du gradient d'un réseau à 2 neurones jusqu'à la mesure du coût (paramètres, FLOPs, latence) d'une grille de CNNs configurables, avec un rapport chiffré par palier. L'objectif est **pédagogique** et prépare un stage de recherche d'architectures neuronales (NAS) : lire et écrire du code PyTorch propre, mesurer honnêtement, rester reproductible.

**Accès rapide** : [Méthode de travail](#méthode-de-travail) · [Parcours](#parcours) · [Démarrage rapide](#démarrage-rapide) · [Résultats](#résultats) · [English summary](#english-summary) · [Limites](#limites)

## Méthode de travail

Je n'ai pas écrit le code de ce dépôt : il est écrit par un assistant IA sous ma direction. Ma part : la conception (ordre des paliers, périmètre, ce que l'on mesure), le protocole, la relecture critique des résultats et la compréhension de chaque mécanisme.

Chaque palier suit un protocole allégé :
1. **Prédiction chiffrée** écrite avant tout entraînement (une prédiction fausse est un résultat utile).
2. **Tests rapides** sur CPU, données synthétiques, sans réseau (formes, gradient, config, FLOPs).
3. **Mesures** sur le jeu de validation ; 1 graine en exploration, 3 graines pour les chiffres retenus ; le jeu test n'est lu qu'à la fin des paliers 3 et 5.
4. **Rapport court** dans [`experiments/`](experiments/), échecs compris.
5. **Porte de compréhension** : trois questions auxquelles je dois savoir répondre sans consulter le dépôt.

## Parcours

Spec de référence : `docs/superpowers/specs/2026-10-12-cnn-pytorch-design.md`.

| N° | Palier | Ce qu'il enseigne | Statut |
|---|---|---|---|
| 1 | Autograd sur un réseau à 2 neurones | graphe de calcul, accumulation de `.grad`, `no_grad`, `detach` | Terminé ([rapport](experiments/s01-autograd.md)) |
| 2 | MLP sur FashionMNIST | `Dataset`/`DataLoader`, `nn.Module`, boucle train/eval, device | Terminé ([rapport](experiments/s02-mlp-fashion.md)) |
| 3 | CNN sur CIFAR-10 | `Conv2d`, `BatchNorm`, formes `(B, C, H, W)`, scheduler, checkpoint | Terminé ([rapport](experiments/s03-cnn-cifar.md)) |
| 4 | CNN configurable depuis un dict | `Sequential`/`ModuleList`, pooling global, validation de config | Terminé ([rapport](experiments/s04-cnn-config.md)) |
| 5 | Coût d'un modèle | paramètres, FLOPs, latence mps et CPU, compromis précision / coût | Terminé ([rapport](experiments/s05-cost.md)) |
| 6 | Random search et front de Pareto | exploration d'architectures et compromis multi-objectifs | Terminé ([rapport](experiments/s06-pareto.md)) |

## Démarrage rapide

Prérequis : Python 3.11 ou plus, [uv](https://docs.astral.sh/uv/). Backend `mps` sur Mac Apple Silicon, repli CPU sinon.

```bash
uv sync
uv run pytest                                  # tests hermétiques, aucun réseau
uv run ruff check .                            # lint
uv run python -m steps.s01_autograd
uv run python -m steps.s02_mlp_fashion --seed 0   # télécharge FashionMNIST dans data/ au premier lancement
uv run python -m steps.s03_cnn_cifar --seed 0     # télécharge CIFAR-10 dans data/
uv run python -m steps.s05_cost --seed 0          # grille de 6 configs : plusieurs dizaines de minutes
uv run python -m common.results s05 config_name val_acc   # synthèse moyenne et écart-type par config
uv run python -m steps.s06_pareto --n-models 15 --epochs 3 --seed 42  # benchmark NAS et frontières de Pareto
```

## Résultats

Tous les résultats ci-dessous sont extraits des rapports de mesure dans `experiments/` et ont été mesurés sur Mac M4 (mps avec repli CPU, Python 3.12, PyTorch 2.14).

### Palier 1 : Autograd sur un micro-réseau à 2 neurones
* Dispositif : architecture h = tanh(w1 * x), y_hat = w2 * h, perte quadratique, x = 0.5, y = 1.0, w1 = 0.8, w2 = -0.4, descente avec lr = 0.1 sur 50 pas.
* Gradients initiaux : concordance exacte à 1e-9 près entre formule analytique et autograd PyTorch (dw1 = 0.394271, dw2 = -0.875387).
* Règle d'accumulation : sans remise à zéro via `optimizer.zero_grad()`, le gradient s'additionne à chaque passe (dw1 = 0.788543 après 2 passes) et empêche toute convergence. Avec remise à zéro, la perte descend de 1.3271 à 0.0004.

### Palier 2 : Perceptron multicouche sur FashionMNIST
* Dispositif : architecture 784 -> 256 -> 128 -> 10 (235 146 paramètres), optimiseur Adam (lr = 1e-3), batch size 128, 10 époques.
* Résultats sur le jeu de validation (5 000 images, 3 graines : 0, 1, 2) :
  * `val_acc` moyenne : **89,61 %** (dispersion ± 0,03 point)
  * `val_loss` moyenne : 0,3053
  * Temps d'entraînement moyen : 22,8 s par graine.

### Palier 3 : CNN de base sur CIFAR-10
* Dispositif : 3 blocs Conv2d(k=3, pad=1)-BatchNorm2d-ReLU-MaxPool2d (32, 64, 128 canaux) suivis d'un AdaptiveAvgPool2d((1, 1)) et d'une couche Linear(128, 10) (94 762 paramètres). Optimiseur AdamW (lr = 3e-3, weight decay = 5e-4), scheduler OneCycleLR, batch size 128, 15 époques, augmentations (RandomCrop, RandomHorizontalFlip).
* Résultats sur 3 graines (0, 1, 2) :
  * Validation (5 000 images) : `val_acc` = **78,19 %** ± 0,32 %, `val_loss` = 0,6233
  * Test (10 000 images, option `--test`) : `test_acc` = **77,60 %** ± 0,08 %, `test_loss` = 0,6500
  * Temps d'entraînement moyen : 101,7 s par graine (~1 min 42 s).

### Palier 4 : CNN configurable (largeur vs profondeur) sur CIFAR-10
* Dispositif : 10 époques, batch size 128, AdamW + OneCycleLR, graine 0 en exploration sur jeu validation (5 000 images).
* Comparatif d'architectures :
  * `s03_equivalent` (baseline, 1 bloc/étage, canaux 32-64-128) : 94 762 paramètres, 46,0 s d'entraînement, `val_acc` = 74,82 %
  * `wide` (canaux doublés 64-128-256, 1 bloc/étage) : 373 834 paramètres, 105,3 s d'entraînement, `val_acc` = 80,06 % (+5,24 points)
  * `deep` (canaux 32-64-128, 2 blocs/étage) : 288 746 paramètres, 155,2 s d'entraînement, `val_acc` = **86,36 %** (+11,54 points)
* Constat : l'approfondissement surpasse nettement l'élargissement en précision (+6,30 points avec 23 % de paramètres en moins), mais s'avère 47 % plus lent par époque à cause du caractère séquentiel des couches qui limite le parallélisme matériel.

### Palier 5 : Mesure du coût (paramètres, FLOPs, latence mps/CPU) sur CIFAR-10
* Dispositif : grille de 6 configurations entraînées sur 4 époques (batch size 128, AdamW + OneCycleLR). Synthèse retenue sur 3 graines (0, 1, 2) avec évaluation sur jeu test (10 000 images). FLOPs et MACs comptés en batch 1 (FlopCounterMode, 1 MAC = 2 FLOPs). Latence médiane mesurée sur 50 passes (warmup 10 passes, synchronisation matérielle explicite).
* Tableau récapitulatif :

| Config | Paramètres | MACs (M) | FLOPs (M) | test_acc (3 graines) | mps b1 (ms) | mps b64 (ms) | cpu b1 (ms) | cpu b64 (ms) | Temps train moy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tiny` | 5 466 | 1,62 | 3,24 | 44,75 % ± 0,93 % | 0,680 ms | 2,456 ms | 0,142 ms | 4,243 ms | 11,2 s |
| `s03` | 94 762 | 10,32 | 20,65 | 67,43 % ± 0,13 % | 0,419 ms | 2,725 ms | 0,258 ms | 10,383 ms | 19,7 s |
| `late_pool` | 94 762 | 24,48 | 48,96 | 64,40 % ± 0,41 % | 0,466 ms | 4,396 ms | 0,346 ms | 17,644 ms | 32,0 s |
| `kernel5` | 260 138 | 28,67 | 57,35 | 71,95 % ± 0,20 % | 0,642 ms | 4,882 ms | 0,377 ms | 14,451 ms | 33,5 s |
| `deep` | 288 746 | 38,63 | 77,27 | **80,95 % ± 0,23 %** | 0,866 ms | 5,238 ms | 0,530 ms | 33,928 ms | 40,9 s |
| `wide` | 373 834 | 39,52 | 79,04 | 71,79 % ± 0,55 % | 0,690 ms | 5,957 ms | 0,325 ms | 26,032 ms | 45,1 s |

* Enseignements clés sur le coût :
  1. **Surcoût de soumission en batch 1** : le CPU est 2 à 5 fois plus rapide que le GPU (mps) sur une seule image (0,142 ms vs 0,680 ms pour `tiny`) car l'overhead Python-Metal domine le calcul.
  2. **Parallélisme en batch 64** : mps reprend l'avantage massif dès que le lot augmente, se montrant 3 à 6 fois plus rapide que le CPU (5,2 ms vs 33,9 ms pour `deep`).
  3. **FLOPs vs latence matérielle** : les FLOPs ne prédisent pas toujours la latence. En batch 64 sur CPU, `wide` est plus rapide que `deep` (26,0 ms vs 33,9 ms) malgré plus de FLOPs, car ses couches plus larges vectorisent mieux et réduisent les accès mémoire.

### Palier 6 : Random Search et Frontières de Pareto multi-objectifs sur CIFAR-10
* Dispositif : exploration aléatoire d'un espace de recherche (2 à 4 étages, canaux 16 à 128, noyaux 3 et 5, 1 ou 2 blocs par étage, pooling libre) sur 15 architectures distinctes entraînées 3 époques chacune (`seed=42`).
* Résultats majeurs :
  * Modèle dominant en précision : `arch_14` (1,79M params, 338 M MACs) atteint **82,36 %** en 3 époques et domine l'ensemble de l'espace.
  * Modèle le plus efficient : `arch_09` (1,48M params, 188 M MACs) atteint **79,36 %** en 99 secondes et domine strictement `arch_02` (79,30 %, 740 M MACs, 315 s de calcul) avec 4 fois moins d'opérations.
  * Frontière de Pareto Précision vs FLOPs : `arch_01` (63,66 %, 5,36 M MACs), `arch_15` (66,84 %, 20,23 M MACs), `arch_09` (79,36 %, 188,32 M MACs), `arch_14` (82,36 %, 338,04 M MACs).
  * Basculement sur petit lot (mps batch 1) : `arch_13`, `arch_03`, `arch_06`, `arch_14`. Les FLOPs échouent à prédire la latence en batch 1 : 3 architectures sur 4 du front théorique sont éliminées sur matériel réel au profit de modèles à plus faible overhead d'ordonnancement.
  * Inefficience du Random Search : 73 % des architectures échantillonnées sont dominées, justifiant le recours à l'Optimisation Bayésienne (BO) en NAS.

## English summary

A six-step PyTorch curriculum building a miniature NAS: autograd on a two-neuron network, an MLP on FashionMNIST, a CNN on CIFAR-10, a CNN built from a config dict, a cost benchmark (parameters, FLOPs, mps and CPU latency) over a grid of six configurations, and a multi-objective random search computing Pareto frontiers. Each step has a short measured write-up (in French), a written prediction made before training, and CPU-only tests. The code is written by an AI assistant under the direction of the repository owner, who designed the curriculum and the protocol and reviews every result.

## Limites

* Deux jeux d'images seulement (FashionMNIST, CIFAR-10) et des modèles très petits : aucune conclusion ne se généralise à de grands modèles.
* `mps` n'est pas déterministe bit-à-bit : deux exécutions de même graine peuvent différer légèrement.
* La latence sur mps en batch 1 est dominée par le coût de soumission des opérations ; elle dit peu sur un autre matériel.
* `FlopCounterMode` ne compte que les produits matriciels et les convolutions (pas les activations, le pooling ni BatchNorm).
* Les entraînements durent quelques époques : les classements de précision entre configs sont indicatifs.
* Un seul auteur de l'évaluation, sans second relecteur.
