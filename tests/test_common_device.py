import json
import random

import numpy as np
import torch

from common.device import environment_info, get_device, set_seed, synchronize


def test_get_device_is_mps_or_cpu():
    assert get_device().type in {"mps", "cpu"}


def test_set_seed_makes_random_numpy_and_torch_reproducible():
    set_seed(3)
    first = (random.random(), np.random.rand(), torch.rand(2))
    set_seed(3)
    second = (random.random(), np.random.rand(), torch.rand(2))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_synchronize_is_a_noop_on_cpu():
    synchronize(torch.device("cpu"))


def test_environment_info_is_json_serializable():
    info = environment_info(torch.device("cpu"))
    assert json.loads(json.dumps(info)) == info
    assert set(info) == {"device", "python", "torch"}
