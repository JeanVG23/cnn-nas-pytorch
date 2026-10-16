"""Palier 5 : mesurer le coût d'un modèle (paramètres, FLOPs, latence) et le comparer à la précision.

Unités : 1 MAC (multiplication-accumulation) = 2 FLOPs. Le compteur de torch ne couvre que les
produits matriciels et les convolutions (pas les activations, le pooling, BatchNorm, ni les biais).
Latence : boucle écrite à la main (warm-up, synchronize avant chaque lecture d'horloge sur mps,
médiane sur plusieurs passes). Sur batch 1, la latence mps est dominée par le coût de soumission des
opérations (Python vers Metal), pas par le calcul : le CPU y est souvent plus rapide.

Usage : uv run python -m steps.s05_cost --seed 0 --epochs 4
"""

import argparse
import statistics
import time

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from common.data import cifar10_loaders, cifar10_test_loader
from common.device import environment_info, get_device, set_seed, synchronize
from common.results import append_result
from common.train import evaluate, fit
from steps.s04_cnn_config import build_model, stage

GRID = {
    "tiny": {"stages": [stage(16), stage(32)]},
    "s03": {"stages": [stage(32), stage(64), stage(128)]},
    "wide": {"stages": [stage(64), stage(128), stage(256)]},
    "deep": {"stages": [stage(32, blocks=2), stage(64, blocks=2), stage(128, blocks=2)]},
    "kernel5": {"stages": [stage(32, kernel=5), stage(64, kernel=5), stage(128, kernel=5)]},
    # mêmes paramètres que s03, mais le deuxième pooling est retardé : plus de FLOPs sans plus de paramètres
    "late_pool": {"stages": [stage(32), stage(64, pool=False), stage(128)]},
}
BATCH_SIZES = (1, 64)
INPUT_SHAPE = (3, 32, 32)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """FLOPs de la passe avant en mode eval (1 MAC = 2 FLOPs), sur le device du modèle."""
    device = next(model.parameters()).device
    model.eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops()


def measure_latency(model: nn.Module, input_shape: tuple[int, ...], device: torch.device, *, warmup: int = 10, runs: int = 50) -> float:
    """Latence médiane d'une passe avant, en millisecondes."""
    model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):  # les premières passes compilent les noyaux et allouent la mémoire : on les écarte
            model(x)
        synchronize(device)
        for _ in range(runs):
            started = time.perf_counter()
            model(x)
            synchronize(device)  # sans cela, sur mps, on ne mesure que la soumission, pas l'exécution
            times.append(time.perf_counter() - started)
    return statistics.median(times) * 1000


def measurement_devices(device: torch.device) -> list[torch.device]:
    """Le device d'entraînement, plus le CPU s'il est différent : la comparaison mps / CPU est le point d'intérêt."""
    cpu = torch.device("cpu")
    return [device] if device.type == "cpu" else [device, cpu]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test", action="store_true", help="lire le jeu test (run final uniquement)")
    args = parser.parse_args()

    device = get_device()
    train_loader, val_loader = cifar10_loaders(args.batch_size)
    test_loader = cifar10_test_loader() if args.test else None

    rows = []
    for name, config in GRID.items():
        print(f"=== {name} ===")
        set_seed(args.seed)  # même graine d'initialisation pour chaque config
        model = build_model(config)
        result = fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

        record = {
            "config_name": name,
            "config": config,
            "seed": args.seed,
            "epochs": args.epochs,
            "n_params": count_params(model),
            "flops_b1": count_flops(model, (1, *INPUT_SHAPE)),
            "val_acc": round(result["val_acc"], 4),
            "train_seconds": round(result["train_seconds"], 2),
            **environment_info(device),
        }
        record["macs_b1"] = record["flops_b1"] // 2
        if test_loader is not None:
            record["test_acc"] = round(evaluate(model, test_loader, device)[1], 4)
        for dev in measurement_devices(device):  # après l'évaluation : measure_latency déplace le modèle
            for batch in BATCH_SIZES:
                record[f"latency_{dev.type}_b{batch}_ms"] = round(measure_latency(model, (batch, *INPUT_SHAPE), dev), 3)
        append_result("s05", record)
        rows.append(record)

    latency_keys = [k for k in rows[0] if k.startswith("latency_")]
    print()
    print(f"{'config':10s} {'params':>9s} {'MACs(M)':>8s} {'FLOPs(M)':>9s} {'val_acc':>8s} " + " ".join(f"{k[8:-3]:>9s}" for k in latency_keys))
    for r in sorted(rows, key=lambda r: r["n_params"]):
        latencies = " ".join(f"{r[k]:9.3f}" for k in latency_keys)
        print(f"{r['config_name']:10s} {r['n_params']:9d} {r['macs_b1'] / 1e6:8.2f} {r['flops_b1'] / 1e6:9.2f} {r['val_acc']:8.4f} {latencies}")
    print("latences en ms (médiane) ; MACs et FLOPs pour une image (batch 1) ; 1 MAC = 2 FLOPs")


if __name__ == "__main__":
    main()
