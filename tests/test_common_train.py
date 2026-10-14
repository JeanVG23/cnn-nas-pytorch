import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.device import set_seed
from common.train import evaluate, fit

CPU = torch.device("cpu")
FIT_KEYS = {"val_acc", "val_loss", "train_loss", "train_seconds", "epoch_seconds", "history"}


def tiny_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


def separable_loaders() -> tuple[DataLoader, DataLoader]:
    """Deux classes faciles à séparer : images proches de -1 (classe 0) ou de +1 (classe 1)."""
    generator = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (96,), generator=generator)
    x = (y.float() * 2 - 1).view(-1, 1, 1, 1) + 0.1 * torch.randn(96, 3, 8, 8, generator=generator)
    train = DataLoader(TensorDataset(x[:64], y[:64]), batch_size=16, shuffle=True)
    val = DataLoader(TensorDataset(x[64:], y[64:]), batch_size=16)
    return train, val


def test_fit_returns_the_documented_contract():
    set_seed(0)
    train, val = separable_loaders()
    result = fit(tiny_cnn(), train, val, epochs=2, lr=1e-2, device=CPU)
    assert set(result) == FIT_KEYS
    assert len(result["epoch_seconds"]) == 2
    assert len(result["history"]) == 2
    assert result["history"][-1]["epoch"] == 2
    assert result["train_seconds"] == pytest.approx(sum(result["epoch_seconds"]))
    assert result["val_acc"] == result["history"][-1]["val_acc"]


def test_fit_learns_a_separable_problem():
    set_seed(0)
    train, val = separable_loaders()
    result = fit(tiny_cnn(), train, val, epochs=8, lr=1e-2, device=CPU)
    assert result["val_acc"] >= 0.9


def test_fit_rejects_zero_epochs():
    train, val = separable_loaders()
    with pytest.raises(ValueError, match="epochs"):
        fit(tiny_cnn(), train, val, epochs=0, lr=1e-2, device=CPU)


def test_evaluate_returns_loss_and_accuracy_in_unit_interval():
    set_seed(0)
    _, val = separable_loaders()
    loss, accuracy = evaluate(tiny_cnn(), val, CPU)
    assert loss > 0
    assert 0.0 <= accuracy <= 1.0
