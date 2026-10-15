"""Palier 4 : un CNN configurable depuis un dict. C'est l'espace de recherche du futur NAS.

Une config est {"stages": [stage, ...]} ; un étage est {"channels", "kernel", "blocks", "pool"}.
Chaque étage empile `blocks` fois (Conv2d sans biais, BatchNorm2d, ReLU), puis un MaxPool2d(2) si
`pool`. La tête est toujours AdaptiveAvgPool2d((1, 1)), Flatten, Linear(canaux du dernier étage, 10) :
sa taille ne dépend jamais de la taille spatiale, donc on peut changer la profondeur librement.

Usage : uv run python -m steps.s04_cnn_config --config wide --seed 0 --epochs 10
"""

import argparse

import torch
from torch import nn

from common.data import cifar10_loaders
from common.device import environment_info, get_device, set_seed
from common.results import append_result
from common.train import fit

INPUT_SIZE = 32
IN_CHANNELS = 3
NUM_CLASSES = 10
STAGE_KEYS = {"channels", "kernel", "blocks", "pool"}


def stage(channels: int, kernel: int = 3, blocks: int = 1, pool: bool = True) -> dict:
    return {"channels": channels, "kernel": kernel, "blocks": blocks, "pool": pool}


CONFIGS = {
    "s03_equivalent": {"stages": [stage(32), stage(64), stage(128)]},
    "wide": {"stages": [stage(64), stage(128), stage(256)]},
    "deep": {"stages": [stage(32, blocks=2), stage(64, blocks=2), stage(128, blocks=2)]},
}


def validate_config(config: dict) -> None:
    """Refuse toute config invalide avec un message clair, avant de construire quoi que ce soit."""
    stages = config.get("stages") if isinstance(config, dict) else None
    if not isinstance(stages, list) or not stages:
        raise ValueError("config['stages'] doit être une liste non vide d'étages")
    size = INPUT_SIZE
    for i, st in enumerate(stages):
        if not isinstance(st, dict) or set(st) != STAGE_KEYS:
            raise ValueError(f"étage {i} : clés attendues {sorted(STAGE_KEYS)}, reçu {st!r}")
        for key in ("channels", "kernel", "blocks"):
            value = st[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"étage {i} : {key} doit être un entier >= 1, reçu {value!r}")
        if st["kernel"] % 2 == 0:
            raise ValueError(
                f"étage {i} : kernel doit être impair pour garder la taille avec padding, reçu {st['kernel']}"
            )
        if not isinstance(st["pool"], bool):
            raise ValueError(f"étage {i} : pool doit être un booléen, reçu {st['pool']!r}")
        if st["pool"]:
            size //= 2
            if size < 1:
                raise ValueError(
                    f"étage {i} : la taille spatiale tombe à 0 (images {INPUT_SIZE}x{INPUT_SIZE} : au plus 5 étages avec pooling)"
                )


def conv_stage(c_in: int, st: dict) -> nn.Sequential:
    layers: list[nn.Module] = []
    for b in range(st["blocks"]):
        layers += [
            nn.Conv2d(
                c_in if b == 0 else st["channels"],
                st["channels"],
                st["kernel"],
                padding=st["kernel"] // 2,
                bias=False,
            ),
            nn.BatchNorm2d(st["channels"]),
            nn.ReLU(),
        ]
    if st["pool"]:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ConfigCNN(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        validate_config(config)
        # nn.ModuleList (et non une liste Python) : sinon les paramètres des étages seraient invisibles pour l'optimiseur
        self.stages = nn.ModuleList()
        c_in = IN_CHANNELS
        for st in config["stages"]:
            self.stages.append(conv_stage(c_in, st))
            c_in = st["channels"]
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(c_in, NUM_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, 3, 32, 32)
        for stage_module in self.stages:
            x = stage_module(x)  # (B, channels, H, W) ; H et W divisés par 2 si pool
        return self.head(x)  # (B, 10)


def build_model(config: dict) -> ConfigCNN:
    return ConfigCNN(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="s03_equivalent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader = cifar10_loaders(args.batch_size)
    config = CONFIGS[args.config]
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} config={args.config} paramètres={n_params}")

    result = fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    record = {
        "config_name": args.config,
        "config": config,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_params": n_params,
        "train_seconds": round(result["train_seconds"], 2),
        "val_loss": round(result["val_loss"], 4),
        "val_acc": round(result["val_acc"], 4),
        **environment_info(device),
    }
    print(append_result("s04", record))


if __name__ == "__main__":
    main()
