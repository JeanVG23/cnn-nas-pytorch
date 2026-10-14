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
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


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


def cifar10_loaders(batch_size: int, data_dir: Path = DATA_DIR) -> tuple[DataLoader, DataLoader]:
    """CIFAR-10 : 45 000 images d'entraînement (avec augmentation), 5 000 de validation (sans). Images (B, 3, 32, 32).

    Deux instances du même jeu avec des transformations différentes, découpées avec les mêmes indices :
    la validation ne doit jamais être augmentée.
    """
    normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    full_eval = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_transform)
    train_idx, val_idx = split_indices(len(full_train), N_VAL)
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_eval, val_idx), batch_size=EVAL_BATCH, shuffle=False)
    return train_loader, val_loader


def cifar10_test_loader(data_dir: Path = DATA_DIR) -> DataLoader:
    """Jeu test de CIFAR-10 (10 000 images). À n'appeler que pour les runs finaux (option --test)."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return DataLoader(test, batch_size=EVAL_BATCH, shuffle=False)

