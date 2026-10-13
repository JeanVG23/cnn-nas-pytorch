"""Données : découpe train/val déterministe et loaders FashionMNIST.

Le jeu test n'est jamais chargé par les fonctions de découpe : il vient de fonctions dédiées.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = Path("data")
SPLIT_SEED = 1234  # propre à la découpe : changer --seed ne change pas le jeu de validation
N_VAL = 5000
EVAL_BATCH = 512

FASHION_MEAN, FASHION_STD = (0.2860,), (0.3530,)


def split_indices(n: int, n_val: int, split_seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
    """Indices (train, val) disjoints, tirés avec une graine propre à la découpe."""
    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n, generator=generator).tolist()
    return perm[n_val:], perm[:n_val]


def fashion_mnist_loaders(batch_size: int, data_dir: Path = DATA_DIR) -> tuple[DataLoader, DataLoader]:
    """FashionMNIST : 55 000 images d'entraînement, 5 000 de validation. Images (B, 1, 28, 28), étiquettes (B,)."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(FASHION_MEAN, FASHION_STD)])
    full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    train_idx, val_idx = split_indices(len(full), N_VAL)
    train_loader = DataLoader(Subset(full, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=EVAL_BATCH, shuffle=False)
    return train_loader, val_loader
