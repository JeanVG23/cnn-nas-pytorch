"""Device (mps ou cpu), graines, synchronisation et informations d'environnement."""

import platform
import random

import numpy as np
import torch


def get_device() -> torch.device:
    """mps (GPU du Mac) si disponible, sinon cpu."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Fixe random, numpy et torch. mps n'est pas déterministe bit-à-bit : deux exécutions de même graine peuvent différer légèrement."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device: torch.device) -> None:
    """Attend la fin des calculs en cours. Sur mps les opérations sont asynchrones : sans cet appel, une horloge ne mesure que le temps de soumission."""
    if device.type == "mps":
        torch.mps.synchronize()


def environment_info(device: torch.device) -> dict:
    """Champs d'environnement ajoutés à chaque ligne JSONL de résultats."""
    return {"device": device.type, "python": platform.python_version(), "torch": torch.__version__}
