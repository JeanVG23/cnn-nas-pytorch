"""Palier 3 : petit CNN sur CIFAR-10.

Trois blocs Conv2d-BatchNorm-ReLU-MaxPool (canaux 32, 64, 128), puis un pooling global
(AdaptiveAvgPool2d) et une couche linéaire : la tête ne dépend pas de la taille spatiale.
On suit la forme (B, C, H, W) à chaque bloc, on utilise un scheduler (dans common/train.py) et on
sauvegarde puis recharge un checkpoint. --test lit le jeu test (à réserver au run final).

Usage : uv run python -m steps.s03_cnn_cifar --seed 0 --epochs 15
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from common.data import cifar10_loaders, cifar10_test_loader
from common.device import environment_info, get_device, set_seed
from common.results import append_result
from common.train import evaluate, fit

CHECKPOINT = Path("data/checkpoints/s03_cnn_cifar.pt")


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),  # (B, c_out, H, W) : le biais est inutile avant BatchNorm
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),  # (B, c_out, H/2, W/2)
    )


class CifarCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 128, 4, 4) -> (B, 128, 1, 1) : indépendant de la taille spatiale
            nn.Flatten(),  # (B, 128)
            nn.Linear(128, 10),  # (B, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, 3, 32, 32)
        return self.head(self.features(x))  # (B, 10)


def trace_shapes(model: CifarCNN, x: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """Forme du tenseur à l'entrée, après chaque bloc et après la tête."""
    shapes = [("input", tuple(x.shape))]
    with torch.no_grad():
        for i, block in enumerate(model.features, start=1):
            x = block(x)
            shapes.append((f"block{i}", tuple(x.shape)))
        shapes.append(("head", tuple(model.head(x).shape)))
    return shapes


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path) -> None:
    """map_location="cpu" : un checkpoint produit sur mps se recharge n'importe où ; load_state_dict copie vers le device du modèle."""
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test", action="store_true", help="lire le jeu test (run final uniquement)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader = cifar10_loaders(args.batch_size)
    model = CifarCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} paramètres={n_params}")

    x, _ = next(iter(val_loader))
    model.eval()  # en mode train, ce passage à vide mettrait à jour les statistiques de BatchNorm avec 2 images
    for name, shape in trace_shapes(model, x[:2].to(device)):
        print(f"  {name:7s} {shape}")

    result = fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    save_checkpoint(model, CHECKPOINT)
    reloaded = CifarCNN().to(device)
    load_checkpoint(reloaded, CHECKPOINT)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        x = x.to(device)
        same = torch.allclose(model(x), reloaded(x), atol=1e-5)
    print(f"checkpoint {CHECKPOINT} rechargé, sorties identiques : {same}")

    record = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_params": n_params,
        "train_seconds": round(result["train_seconds"], 2),
        "epoch_seconds": [round(s, 2) for s in result["epoch_seconds"]],
        "val_loss": round(result["val_loss"], 4),
        "val_acc": round(result["val_acc"], 4),
        **environment_info(device),
    }
    if args.test:
        test_loss, test_acc = evaluate(model, cifar10_test_loader(), device)
        record.update({"test_loss": round(test_loss, 4), "test_acc": round(test_acc, 4)})
        print(f"test_acc={test_acc:.4f}")
    print(append_result("s03", record))


if __name__ == "__main__":
    main()
