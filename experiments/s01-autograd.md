# Palier 1 : autograd sur un réseau à 2 neurones

Rapport du 12 octobre 2026. Code : `steps/s01_autograd.py`.

## 1. Question

Les gradients calculés par PyTorch via `loss.backward()` correspondent-ils au calcul manuel par la règle de la chaîne, et que produit l'accumulation sans remise à zéro ?

## 2. Prédiction (écrite par Jean avant toute mesure, jamais modifiée après)

Ca devrait etre la meme chose

## 3. Dispositif

* Modèle : micro-réseau à 2 neurones (h = tanh(w1 * x), y_hat = w2 * h, loss = (y_hat - y)**2)
* Entrée et cible : x = 0.5, y = 1.0
* Poids initiaux : w1 = 0.8, w2 = -0.4
* Descente : lr = 0.1, 50 pas

## 4. Résultats

* Gradients au point initial (x = 0.5, y = 1.0, w1 = 0.8, w2 = -0.4) :
  - Calcul manuel : dw1 = 0.394271, dw2 = -0.875387
  - Autograd : dw1 = 0.394271, dw2 = -0.875387
  - Concordance exacte vérifiée à 1e-9 près.
* Nœuds du graphe identifiés (7 nœuds, de la loss vers les feuilles) :
  `PowBackward0`, `SubBackward0`, `MulBackward0`, `TanhBackward0`, `MulBackward0`, `AccumulateGrad` (pour w1), `AccumulateGrad` (pour w2).
* Décrochage du graphe : `h.requires_grad` vaut `True`, tandis que `h.detach().requires_grad` vaut `False`.
* Accumulation des gradients :
  - 1 backward : dw1 = 0.394271
  - 2 backwards sans remise à zéro : dw1 = 0.788543 (exactement le double).
* Descente de gradient (lr = 0.1, 50 pas) : loss 1.3271 -> 0.0004.

## 5. Surprises et échecs

* Aucune surprise sur la concordance numérique : la formule manuelle et l'autograd coïncident parfaitement grâce au type `torch.float64` aligné sur les flottants 64 bits de Python.
* Sans remise à zéro de `.grad`, la descente ne converge pas : la loss oscille de façon chaotique entre 0.003 et 1.3 (la valeur de 0.0236 obtenue à 50 pas n'est qu'un creux temporaire au milieu de ces oscillations).
* Sensibilité au taux d'apprentissage : avec lr = 2.0, les pas sont trop grands et la descente diverge immédiatement (loss atteignant environ 4e38).
* Présence de deux nœuds `AccumulateGrad` terminaux dans le graphe : chaque paramètre feuille requérant un gradient (`w1` et `w2`) possède son propre opérateur d'accumulation.
