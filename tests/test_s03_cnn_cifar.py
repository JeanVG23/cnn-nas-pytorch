import torch

from common.device import get_device, set_seed
from steps.s03_cnn_cifar import CifarCNN, load_checkpoint, save_checkpoint, trace_shapes


def test_cnn_output_shape():
    out = CifarCNN().eval()(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_cnn_parameter_count():
    # conv 864 + 18432 + 73728 ; BatchNorm 64 + 128 + 256 ; linear 128*10+10
    assert sum(p.numel() for p in CifarCNN().parameters()) == 94_762


def test_trace_shapes_follow_the_spatial_halving():
    shapes = trace_shapes(CifarCNN().eval(), torch.randn(2, 3, 32, 32))
    assert shapes == [
        ("input", (2, 3, 32, 32)),
        ("block1", (2, 32, 16, 16)),
        ("block2", (2, 64, 8, 8)),
        ("block3", (2, 128, 4, 4)),
        ("head", (2, 10)),
    ]


def test_checkpoint_roundtrip_gives_identical_outputs(tmp_path):
    set_seed(0)
    model = CifarCNN().eval()
    x = torch.randn(2, 3, 32, 32)
    path = tmp_path / "ckpt" / "model.pt"
    save_checkpoint(model, path)
    set_seed(1)
    other = CifarCNN().eval()
    assert not torch.allclose(model(x), other(x))  # deux initialisations différentes
    load_checkpoint(other, path)
    assert torch.allclose(model(x), other(x))


def test_checkpoint_saved_from_the_best_device_loads_on_cpu(tmp_path):
    """Review Focus : un checkpoint produit sur mps doit se recharger sur CPU avec les mêmes sorties."""
    set_seed(0)
    model = CifarCNN().to(get_device()).eval()
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)
    x = torch.randn(2, 3, 32, 32)
    expected = model(x.to(get_device())).cpu()
    reloaded = CifarCNN().eval()
    load_checkpoint(reloaded, path)
    assert torch.allclose(reloaded(x), expected, atol=1e-4)
