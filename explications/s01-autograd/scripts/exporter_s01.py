"""Exporte les vraies valeurs PyTorch de l'étape 1 vers donnees/s01.js (chargé par la page, sans serveur).

Réutilise steps/s01_autograd.py sans le modifier.
Usage (depuis la racine du projet) : uv run python explications/s01-autograd/scripts/exporter_s01.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from steps.s01_autograd import (  # noqa: E402
    W1_INIT,
    W2_INIT,
    X,
    Y,
    _parameters,
    accumulated_w1_grad,
    autograd_grads,
    descend,
    forward,
    graph_nodes,
    manual_grads,
)


def tree(fn):
    """Arbre des grad_fn : {name, next: [sous-arbres ou None]} (None = constante Python, sans arête)."""
    if fn is None:
        return None
    return {"name": type(fn).__name__, "next": [tree(n) for n, _ in fn.next_functions]}


def descend_no_zero(lr, steps):
    """Même descente que steps.s01_autograd.descend, mais SANS remise à zéro de .grad (le bug à montrer)."""
    w1, w2 = _parameters(W1_INIT, W2_INIT)
    losses = []
    for _ in range(steps):
        _, _, loss = forward(w1, w2)
        loss.backward()
        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad
        losses.append(loss.item())
    return losses, w1.item(), w2.item()


def error_of(fn):
    try:
        fn()
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


data = {"x": X, "y": Y, "w1": W1_INIT, "w2": W2_INIT}
data["source"] = {
    "file": "steps/s01_autograd.py",
    "lines": (ROOT / "steps" / "s01_autograd.py")
    .read_text(encoding="utf-8")
    .splitlines(),
}

# Point initial
w1, w2 = _parameters(W1_INIT, W2_INIT)
h, y_hat, loss = forward(w1, w2)
data["forward"] = {"h": h.item(), "y_hat": y_hat.item(), "loss": loss.item()}
data["manual"] = list(manual_grads(W1_INIT, W2_INIT))
data["autograd"] = list(autograd_grads(W1_INIT, W2_INIT))
data["graphNodes"] = graph_nodes(loss)
data["graphTree"] = tree(loss.grad_fn)
data["flags"] = {
    "hRequiresGrad": h.requires_grad,
    "hGradFn": type(h.grad_fn).__name__,
    "hDetachRequiresGrad": h.detach().requires_grad,
    "w1IsLeaf": w1.is_leaf,
    "w1GradFn": None if w1.grad_fn is None else type(w1.grad_fn).__name__,
    "lossGradFn": type(loss.grad_fn).__name__,
}

# float32 : l'écart avec la formule à la main
w1_32 = torch.tensor(W1_INIT, dtype=torch.float32, requires_grad=True)
w2_32 = torch.tensor(W2_INIT, dtype=torch.float32, requires_grad=True)
_, _, loss32 = forward(w1_32, w2_32)
loss32.backward()
data["float32"] = [w1_32.grad.item(), w2_32.grad.item()]

# Accumulation de .grad
data["accumulated"] = [
    accumulated_w1_grad(W1_INIT, W2_INIT, backwards=n) for n in range(1, 6)
]
w1b, w2b = _parameters(W1_INIT, W2_INIT)
_, _, lossb = forward(w1b, w2b)
lossb.backward()
data["secondBackwardSameGraph"] = error_of(lambda: lossb.backward())

# no_grad et detach
with torch.no_grad():
    h_ng, y_ng, loss_ng = forward(*_parameters(W1_INIT, W2_INIT))
data["noGrad"] = {
    "hRequiresGrad": h_ng.requires_grad,
    "hGradFn": None if h_ng.grad_fn is None else type(h_ng.grad_fn).__name__,
    "backwardError": error_of(lambda: loss_ng.backward()),
}
w1d, w2d = _parameters(W1_INIT, W2_INIT)
hd = torch.tanh(w1d * X).detach()
(((w2d * hd) - Y) ** 2).backward()
data["detach"] = {
    "w1Grad": None if w1d.grad is None else w1d.grad.item(),
    "w2Grad": w2d.grad.item(),
}

# Mise à jour hors no_grad
w1u, w2u = _parameters(W1_INIT, W2_INIT)
_, _, lossu = forward(w1u, w2u)
lossu.backward()


def inplace_update():
    w1u.__isub__(0.1 * w1u.grad)


data["updateOutsideNoGrad"] = error_of(inplace_update)
w1v = w1u - 0.1 * w1u.grad
lossv = forward(w1v, w2u)[2]
lossv.backward()
data["rebind"] = {
    "isLeaf": w1v.is_leaf,
    "gradFn": type(w1v.grad_fn).__name__,
    "gradAfterBackward": w1v.grad,
}

# Descentes
lrs = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
data["descent"] = {str(lr): descend(W1_INIT, W2_INIT, lr, 50) for lr in lrs}
data["descentNoZero"] = {}
for lr in lrs:
    losses, a, b = descend_no_zero(lr, 50)
    data["descentNoZero"][str(lr)] = {"losses": losses, "w1": a, "w2": b}

# Points de test pour vérifier la page (gradients PyTorch à des points aléatoires)
rng = random.Random(7)
tests = []
for _ in range(6):
    p = {
        k: round(round(rng.uniform(-1.5, 1.5) / 0.05) * 0.05, 2)
        for k in ("x", "y", "w1", "w2")
    }  # grille des curseurs de la page
    a1 = torch.tensor(p["w1"], dtype=torch.float64, requires_grad=True)
    a2 = torch.tensor(p["w2"], dtype=torch.float64, requires_grad=True)
    _, _, lo = forward(a1, a2, p["x"], p["y"])
    lo.backward()
    tests.append({**p, "loss": lo.item(), "dw1": a1.grad.item(), "dw2": a2.grad.item()})
data["tests"] = tests

out = Path(__file__).resolve().parents[1] / "donnees" / "s01.js"
out.write_text(
    "window.S01 = " + json.dumps(data, indent=1, ensure_ascii=False) + ";\n",
    encoding="utf-8",
)
print(f"écrit : {out}")
