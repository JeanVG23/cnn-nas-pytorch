import pytest
import torch

from steps.s03_cnn_cifar import CifarCNN
from steps.s04_cnn_config import CONFIGS, build_model, stage, validate_config


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_s03_equivalent_config_matches_the_s03_cnn():
    built = build_model(CONFIGS["s03_equivalent"])
    reference = CifarCNN()
    assert n_params(built) == n_params(reference) == 94_762
    assert [p.shape for p in built.parameters()] == [p.shape for p in reference.parameters()]


def test_hand_computed_parameter_count_of_a_tiny_config():
    # conv 3*8*3*3 = 216 ; BatchNorm 2*8 = 16 ; linear 8*10+10 = 90
    assert n_params(build_model({"stages": [stage(8, pool=False)]})) == 322


@pytest.mark.parametrize(
    "config",
    [
        CONFIGS["s03_equivalent"],
        CONFIGS["wide"],
        CONFIGS["deep"],
        {"stages": [stage(8, kernel=5), stage(8, kernel=7, blocks=2, pool=False), stage(16)]},
        {"stages": [stage(4, pool=False)]},
    ],
)
def test_every_valid_config_maps_a_cifar_batch_to_ten_logits(config):
    out = build_model(config).eval()(torch.randn(3, 3, 32, 32))
    assert out.shape == (3, 10)


def test_five_pooled_stages_reach_one_by_one_and_batch_of_one_works_in_eval():
    """Review Focus : 5 pooling donnent une taille spatiale 1x1 ; BatchNorm en eval avec un lot de 1 ne doit pas échouer."""
    config = {"stages": [stage(4) for _ in range(5)]}
    model = build_model(config).eval()
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 10)


def test_six_pooled_stages_are_rejected_before_construction():
    with pytest.raises(ValueError, match="taille spatiale"):
        build_model({"stages": [stage(4) for _ in range(6)]})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "liste non vide"),
        ({"stages": []}, "liste non vide"),
        ({"stages": "pas une liste"}, "liste non vide"),
        ("pas un dict", "liste non vide"),
        ({"stages": [{"channels": 8, "kernel": 3, "blocks": 1}]}, "clés attendues"),
        ({"stages": [{**stage(8), "extra": 1}]}, "clés attendues"),
        ({"stages": [stage(8, kernel=4)]}, "kernel"),
        ({"stages": [stage(0)]}, "channels"),
        ({"stages": [stage(8, blocks=0)]}, "blocks"),
        ({"stages": [stage(True)]}, "channels"),
        ({"stages": [stage("8")]}, "channels"),
        ({"stages": [{**stage(8), "pool": "oui"}]}, "pool"),
    ],
)
def test_malformed_configs_raise_a_clear_value_error(config, message):
    with pytest.raises(ValueError, match=message):
        validate_config(config)
