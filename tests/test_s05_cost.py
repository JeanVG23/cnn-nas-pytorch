import torch
from torch import nn

from steps.s05_cost import (
    GRID,
    count_flops,
    count_params,
    measure_latency,
    measurement_devices,
)

CPU = torch.device("cpu")


def test_count_params():
    assert count_params(nn.Linear(10, 5)) == 55


def test_flops_of_a_linear_layer_follow_two_times_in_times_out_times_batch():
    assert count_flops(nn.Linear(16, 8), (4, 16)) == 2 * 16 * 8 * 4


def test_flops_of_a_convolution_follow_the_textbook_formula():
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    batch, c_in, c_out, k, h, w = 2, 3, 8, 3, 16, 16
    assert count_flops(conv, (batch, c_in, h, w)) == 2 * c_out * c_in * k * k * h * w * batch


def test_flop_counter_ignores_activations_and_pooling():
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    with_extras = nn.Sequential(conv, nn.ReLU(), nn.MaxPool2d(2))
    assert count_flops(with_extras, (2, 3, 16, 16)) == count_flops(conv, (2, 3, 16, 16))


def test_latency_is_positive():
    assert measure_latency(nn.Linear(8, 8), (1, 8), CPU, warmup=1, runs=3) > 0


def test_measurement_devices_cpu_only():
    """Review Focus : sans mps, on ne mesure que le CPU."""
    assert measurement_devices(CPU) == [CPU]


def test_measurement_devices_adds_cpu_after_an_accelerator():
    devices = measurement_devices(torch.device("mps"))
    assert [d.type for d in devices] == ["mps", "cpu"]


def test_grid_has_six_distinct_configs_that_all_build():
    from steps.s04_cnn_config import build_model

    assert len(GRID) == 6
    for config in GRID.values():
        assert build_model(config).eval()(torch.randn(1, 3, 32, 32)).shape == (1, 10)
