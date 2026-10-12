"""Palier 1 : autograd sur un réseau à 2 neurones.

Réseau : h = tanh(w1 * x), y_hat = w2 * h, loss = (y_hat - y) ** 2.
On compare le gradient écrit à la main (déjà connu depuis mini_llm) à loss.backward(), puis on
regarde ce qui est propre à PyTorch : le graphe de calcul (grad_fn), le piège de l'accumulation
de .grad, torch.no_grad() et .detach().

Usage : uv run python -m steps.s01_autograd
"""

import argparse
import math

import torch

X, Y = 0.5, 1.0
W1_INIT, W2_INIT = 0.8, -0.4


def forward(w1: torch.Tensor, w2: torch.Tensor, x: float = X, y: float = Y) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Passe avant. Chaque opération sur un tenseur qui requiert un gradient ajoute un noeud au graphe."""
    h = torch.tanh(w1 * x)
    y_hat = w2 * h
    loss = (y_hat - y) ** 2
    return h, y_hat, loss


def manual_grads(w1: float, w2: float, x: float = X, y: float = Y) -> tuple[float, float]:
    """Gradient de la loss par rapport à w1 et w2, par la règle de la chaîne, en Python pur."""
    h = math.tanh(w1 * x)
    y_hat = w2 * h
    d_y_hat = 2 * (y_hat - y)  # dloss / dy_hat
    dw2 = d_y_hat * h  # dy_hat / dw2 = h
    dw1 = d_y_hat * w2 * (1 - h**2) * x  # dy_hat / dh = w2, dh / dw1 = (1 - tanh^2) * x
    return dw1, dw2


def _parameters(w1_value: float, w2_value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Deux feuilles du graphe en float64 : requires_grad=True demande à torch de suivre toutes les opérations qui les touchent."""
    w1 = torch.tensor(w1_value, dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(w2_value, dtype=torch.float64, requires_grad=True)
    return w1, w2


def autograd_grads(w1_value: float, w2_value: float) -> tuple[float, float]:
    w1, w2 = _parameters(w1_value, w2_value)
    _, _, loss = forward(w1, w2)
    loss.backward()
    return w1.grad.item(), w2.grad.item()


def accumulated_w1_grad(w1_value: float, w2_value: float, backwards: int) -> float:
    """Le piège : backward() AJOUTE au .grad existant. Sans remise à zéro, deux appels donnent le double."""
    w1, w2 = _parameters(w1_value, w2_value)
    for _ in range(backwards):
        _, _, loss = forward(w1, w2)
        loss.backward()
    return w1.grad.item()


def graph_nodes(loss: torch.Tensor) -> list[str]:
    """Noms des noeuds du graphe, de la loss vers les feuilles (parcours en profondeur sur grad_fn.next_functions)."""
    seen, names, stack = set(), [], [loss.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        names.append(type(fn).__name__)
        stack.extend(next_fn for next_fn, _ in fn.next_functions)
    return names


def descend(w1_value: float, w2_value: float, lr: float, steps: int) -> list[float]:
    """Descente de gradient. La mise à jour se fait sous no_grad : elle ne doit pas entrer dans le graphe."""
    w1, w2 = _parameters(w1_value, w2_value)
    losses = []
    for _ in range(steps):
        _, _, loss = forward(w1, w2)
        loss.backward()
        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad
        w1.grad = None  # remise à zéro : sinon le prochain backward() s'additionnerait
        w2.grad = None
        losses.append(loss.item())
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    dw1, dw2 = manual_grads(W1_INIT, W2_INIT)
    a_dw1, a_dw2 = autograd_grads(W1_INIT, W2_INIT)
    print(f"gradient à la main : dw1={dw1:.6f} dw2={dw2:.6f}")
    print(f"gradient autograd  : dw1={a_dw1:.6f} dw2={a_dw2:.6f}")

    w1, w2 = _parameters(W1_INIT, W2_INIT)
    h, _, loss = forward(w1, w2)
    print(f"noeuds du graphe   : {graph_nodes(loss)}")
    print(f"h requiert un gradient : {h.requires_grad} ; h.detach() : {h.detach().requires_grad}")
    print(f".grad de w1 après 1 backward : {a_dw1:.6f}")
    print(f".grad de w1 après 2 backward sans remise à zéro : {accumulated_w1_grad(W1_INIT, W2_INIT, backwards=2):.6f}")

    losses = descend(W1_INIT, W2_INIT, args.lr, args.steps)
    print(f"descente : loss {losses[0]:.4f} -> {losses[-1]:.4f} en {args.steps} pas (lr={args.lr})")


if __name__ == "__main__":
    main()
