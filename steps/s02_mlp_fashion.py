"""Palier 2 : MLP sur FashionMNIST.

Premier entraînement complet en PyTorch : Dataset/DataLoader, nn.Module, nn.Linear, perte,
optimiseur, boucle train/eval, device. La boucle est écrite ici en entier ; à partir du palier 3
elle vit dans common/train.py (ce palier reste figé avec sa version).

Usage : uv run python -m steps.s02_mlp_fashion --seed 0 --epochs 10
"""

import argparse
import time

import torch
from torch import nn

from common.data import fashion_mnist_loaders
from common.device import environment_info, get_device, set_seed
from common.results import append_result


class MLP(nn.Module):
    """784 -> 256 -> 128 -> 10 : l'image aplatie traverse deux couches cachées."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(784, 256),  # (B, 256)
            nn.ReLU(),
            nn.Linear(256, 128),  # (B, 128)
            nn.ReLU(),
            nn.Linear(128, 10),  # (B, 10) : un logit par classe
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, 1, 28, 28)
        return self.net(x)


def train_one_epoch(
    model: nn.Module, loader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """Une époque d'entraînement. Renvoie la loss moyenne sur l'époque."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:  # x (B, 1, 28, 28), y (B,)
        x, y = x.to(device), y.to(device)
        logits = model(x)  # (B, 10)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    """Loss moyenne et précision sur un loader, sans calcul de gradient."""
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # (B, 10)
        total += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader = fashion_mnist_loaders(args.batch_size)
    model = MLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={device} paramètres={n_params}")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"epoch {epoch:2d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    elapsed = time.perf_counter() - started

    result = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_params": n_params,
        "train_seconds": round(elapsed, 2),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        **environment_info(device),
    }
    print(append_result("s02", result))


if __name__ == "__main__":
    main()
