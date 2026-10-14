"""Boucle d'entraînement et d'évaluation partagée par les paliers 3 à 5.

Écrite en entier dans steps/s02_mlp_fashion.py ; extraite ici à partir du palier 3.
Le contrat de fit (hyperparamètres explicites, dict de métriques renvoyé) est fixé une fois pour
toutes : le palier 5 entraîne 6 configs avec ce même fit, sans rien modifier ici.
"""

import time

import torch
from torch import nn

from common.device import synchronize


def _train_one_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total, n = torch.zeros((), device=device), 0
    for x, y in loader:  # x (B, C, H, W), y (B,)
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.detach() * len(y)  # on garde la somme sur le device : un seul .item() par époque
        n += len(y)
    return total.item() / n


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Loss moyenne et précision sur un loader, sans calcul de gradient (mode eval : BatchNorm utilise ses statistiques apprises)."""
    model.eval()
    loss_sum = nn.CrossEntropyLoss(reduction="sum")
    total, correct, n = torch.zeros((), device=device), torch.zeros((), device=device), 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # (B, num_classes)
        total += loss_sum(logits, y)
        correct += (logits.argmax(dim=1) == y).sum()
        n += len(y)
    return total.item() / n, correct.item() / n


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    weight_decay: float = 5e-4,
) -> dict:
    """Entraîne model avec AdamW et un scheduler OneCycleLR (lr est le pic), évalue sur val_loader à chaque époque.

    Renvoie {"val_acc", "val_loss", "train_loss", "train_seconds", "epoch_seconds", "history"} :
    val_* et train_loss sont ceux de la dernière époque ; epoch_seconds donne le temps d'entraînement
    de chaque époque (évaluation exclue) ; history est une liste de dicts par époque.
    """
    if epochs < 1:
        raise ValueError(f"epochs doit être >= 1, reçu {epochs}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()

    history, epoch_seconds = [], []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = _train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        synchronize(device)  # sur mps, attendre la fin des calculs avant d'arrêter l'horloge
        epoch_seconds.append(time.perf_counter() - started)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({epoch_seconds[-1]:.1f}s)")

    return {
        "val_acc": val_acc,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "train_seconds": sum(epoch_seconds),
        "epoch_seconds": epoch_seconds,
        "history": history,
    }
