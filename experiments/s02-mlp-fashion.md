# Palier 2 : MLP sur FashionMNIST

Rapport du 12 octobre 2026. Code : `steps/s02_mlp_fashion.py`.

## 1. Question

Quelle précision de validation un perceptron multicouche à deux couches cachées (784 -> 256 -> 128 -> 10) atteint-il sur FashionMNIST après 10 époques avec Adam (lr=1e-3, batch_size=128), et combien de paramètres compte ce réseau ?

## 2. Prédiction (écrite par Jean avant toute mesure, jamais modifiée après)

10%, il compte 250k environs parametres

## 3. Dispositif

* Paramètres : architecture 784 -> 256 -> 128 -> 10, Adam (lr=1e-3), 10 époques, batch_size=128
* Graines : 0 (exploration), 0, 1, 2 (chiffres retenus)
* Device et versions : mps (Mac M4), Python 3.12.13, PyTorch 2.14.1
* Nombre de paramètres : 235 146 (234 624 poids et 522 biais)
* Temps d'entraînement : 22.8 s en moyenne par graine (total ~68 s sur 3 graines)

## 4. Résultats

* Exploration (1 graine, graine 0) : `val_loss = 0.3029`, `val_acc = 89.64 %`
* Synthèse sur 3 graines (jeu `val`, 5 000 images) :
  * Graine 0 : `val_acc = 89.64 %`, `val_loss = 0.3029`
  * Graine 1 : `val_acc = 89.58 %`, `val_loss = 0.3054`
  * Graine 2 : `val_acc = 89.62 %`, `val_loss = 0.3076`
  * Moyenne retenue (3 graines) : `val_acc = 89.61 %` (dispersion minime de +-0.03 point), `val_loss = 0.3053`

## 5. Surprises et échecs

* Paramètres : l'estimation de Jean (~250k) est très proche de la réalité (235 146).
* Précision : la prédiction initiale de 10 % correspondait en réalité au niveau du pur hasard (1 chance sur 10 sans apprentissage). En pratique, le MLP dépasse 86 % dès la première époque et atteint 89.61 % en 10 époques. Même sans convolution, les corrélations de pixels bruts suffisent à discriminer une grande partie des vêtements.
* Exécution : le lancement initial via `python steps/s02_mlp_fashion.py` a échoué sur `ModuleNotFoundError: No module named 'common'`. Résolu en exécutant via le module `python -m steps.s02_mlp_fashion` pour garantir que la racine du projet est présente dans `sys.path`.
