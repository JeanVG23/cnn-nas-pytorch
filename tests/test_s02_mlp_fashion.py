import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.device import set_seed
from steps.s02_mlp_fashion import MLP, evaluate, train_one_epoch

CPU = torch.device("cpu")


def synthetic_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 1, 28, 28, generator=generator)
    y = torch.randint(0, 10, (64,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=16)


def test_mlp_output_shape():
    out = MLP()(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_mlp_parameter_count():
    # 784*256+256 + 256*128+128 + 128*10+10
    assert sum(p.numel() for p in MLP().parameters()) == 235_146


def test_training_on_a_small_synthetic_set_reduces_the_loss():
    set_seed(0)
    model, loader, loss_fn = MLP(), synthetic_loader(), nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before, _ = evaluate(model, loader, loss_fn, CPU)
    for _ in range(30):
        train_one_epoch(model, loader, loss_fn, optimizer, CPU)
    after, _ = evaluate(model, loader, loss_fn, CPU)
    assert after < before


def test_evaluate_returns_loss_and_accuracy_in_unit_interval():
    set_seed(0)
    loss, accuracy = evaluate(MLP(), synthetic_loader(), nn.CrossEntropyLoss(), CPU)
    assert loss > 0
    assert 0.0 <= accuracy <= 1.0
