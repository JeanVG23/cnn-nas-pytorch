"""Palier 6 : Random Search dans un espace de recherche de CNNs et frontière de Pareto.

Explore un espace d'architectures candidates (profondeur, canaux, taille de noyaux, pooling),
évalue chaque candidat sur CIFAR-10 (précision, FLOPs/MACs, latences mps et CPU), et calcule
la frontière de Pareto multi-objectifs pour isoler les architectures optimales sans compromis inutile.
Compare notamment la frontière théorique (Précision vs FLOPs) à la frontière réelle (Précision vs Latence).

Usage : uv run python -m steps.s06_pareto --n-models 15 --epochs 3 --seed 42
"""

import argparse
import random

import torch

from common.data import cifar10_loaders, cifar10_test_loader
from common.device import environment_info, get_device, set_seed
from common.results import append_result
from common.train import evaluate, fit
from steps.s04_cnn_config import build_model, stage, validate_config
from steps.s05_cost import INPUT_SHAPE, count_flops, count_params, measure_latency

CHANNELS_POOL = (16, 32, 64, 128)
KERNELS_POOL = (3, 5)
BLOCKS_POOL = (1, 2)


def sample_config(rng: random.Random) -> dict:
    """Tire aléatoirement une configuration valide de CNN dans l'espace de recherche.

    Contraintes assurées :
    - 2 à 4 étages
    - Canaux croissants ou constants (progression pyramidale standard)
    - Taille spatiale minimale >= 2x2 (évite toute division à 0)
    """
    n_stages = rng.choice([2, 3, 4])
    stages = []
    c_prev_idx = rng.randint(0, len(CHANNELS_POOL) - 1)
    size = 32

    for i in range(n_stages):
        c_idx = rng.randint(c_prev_idx, len(CHANNELS_POOL) - 1)
        channels = CHANNELS_POOL[c_idx]
        c_prev_idx = c_idx

        kernel = rng.choice(KERNELS_POOL)
        blocks = rng.choice(BLOCKS_POOL)

        # On n'applique le pooling que si la résolution spatiale résultante reste >= 2
        can_pool = (size // 2) >= 2
        if i == n_stages - 1 and size > 2:
            pool = True
        elif can_pool:
            pool = rng.choice([True, False])
        else:
            pool = False

        if pool:
            size //= 2

        stages.append(stage(channels=channels, kernel=kernel, blocks=blocks, pool=pool))

    cfg = {"stages": stages}
    validate_config(cfg)
    return cfg


def generate_search_space(n_samples: int, seed: int = 42) -> list[dict]:
    """Génère n_samples configurations d'architectures distinctes de manière déterministe."""
    rng = random.Random(seed)
    configs: list[dict] = []
    seen: set[tuple] = set()
    attempts = 0
    max_attempts = n_samples * 100

    while len(configs) < n_samples and attempts < max_attempts:
        attempts += 1
        cfg = sample_config(rng)
        signature = tuple(
            (st["channels"], st["kernel"], st["blocks"], st["pool"])
            for st in cfg["stages"]
        )
        if signature not in seen:
            seen.add(signature)
            configs.append(cfg)

    if len(configs) < n_samples:
        raise RuntimeError(
            f"Impossible de générer {n_samples} configurations distinctes après {max_attempts} essais"
        )
    return configs


def summarize_config(cfg: dict) -> str:
    """Représentation compacte et lisible d'une architecture."""
    parts = []
    for st in cfg["stages"]:
        p = "P" if st["pool"] else "-"
        parts.append(f"{st['blocks']}x{st['channels']}k{st['kernel']}{p}")
    return " > ".join(parts)


def is_dominated(candidate: dict, population: list[dict], objectives: list[tuple[str, str]]) -> bool:
    """Vérifie si candidate est dominé au sens de Pareto par au moins un membre de population.

    objectives : liste de tuples (clé_métrique, direction) où direction est 'max' ou 'min'.
    candidate est dominé par other s'il existe other dans population tel que :
      1. other est au moins aussi bon que candidate sur tous les objectifs
      2. other est strictement meilleur que candidate sur au moins un objectif.
    """
    for _, direction in objectives:
        if direction not in ("max", "min"):
            raise ValueError(f"direction inconnue: {direction!r} (attendu 'max' ou 'min')")

    for other in population:
        if other is candidate:
            continue
        at_least_as_good = True
        strictly_better = False
        for key, direction in objectives:
            val_other = other[key]
            val_cand = candidate[key]
            if direction == "max":
                if val_other < val_cand:
                    at_least_as_good = False
                    break
                if val_other > val_cand:
                    strictly_better = True
            elif direction == "min":
                if val_other > val_cand:
                    at_least_as_good = False
                    break
                if val_other < val_cand:
                    strictly_better = True
        if at_least_as_good and strictly_better:
            return True
    return False


def compute_pareto_front(records: list[dict], objectives: list[tuple[str, str]]) -> list[dict]:
    """Retourne la liste des éléments de records non dominés selon les objectifs donnés."""
    return [r for r in records if not is_dominated(r, records, objectives)]


def evaluate_architecture(
    name: str,
    config: dict,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    """Profilage des coûts, entraînement proxy et mesure des latences pour une architecture."""
    set_seed(seed)
    model = build_model(config)

    n_params = count_params(model)
    flops_b1 = count_flops(model, (1, *INPUT_SHAPE))
    macs_b1 = flops_b1 // 2

    # Entraînement proxy sur CIFAR-10
    result = fit(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    # Mesures de latence
    cpu = torch.device("cpu")
    latency_cpu_b1 = round(measure_latency(model, (1, *INPUT_SHAPE), cpu, warmup=5, runs=20), 3)
    latency_cpu_b64 = round(measure_latency(model, (64, *INPUT_SHAPE), cpu, warmup=5, runs=20), 3)

    record = {
        "model_id": name,
        "config_summary": summarize_config(config),
        "config": config,
        "seed": seed,
        "epochs": epochs,
        "n_params": n_params,
        "flops_b1": flops_b1,
        "macs_b1": macs_b1,
        "val_acc": round(result["val_acc"], 4),
        "val_loss": round(result["val_loss"], 4),
        "train_seconds": round(result["train_seconds"], 2),
        "latency_cpu_b1_ms": latency_cpu_b1,
        "latency_cpu_b64_ms": latency_cpu_b64,
        **environment_info(device),
    }

    if device.type == "mps":
        record["latency_mps_b1_ms"] = round(measure_latency(model, (1, *INPUT_SHAPE), device, warmup=5, runs=20), 3)
        record["latency_mps_b64_ms"] = round(measure_latency(model, (64, *INPUT_SHAPE), device, warmup=5, runs=20), 3)

    if test_loader is not None:
        record["test_acc"] = round(evaluate(model, test_loader, device)[1], 4)

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-models", type=int, default=15, help="nombre d'architectures échantillonnées (défaut: 15)")
    parser.add_argument("--epochs", type=int, default=3, help="époques d'entraînement proxy par modèle (défaut: 3)")
    parser.add_argument("--seed", type=int, default=42, help="graine de tirage de l'espace de recherche (défaut: 42)")
    parser.add_argument("--train-seed", type=int, default=0, help="graine d'initialisation des poids (défaut: 0)")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test", action="store_true", help="évaluer aussi sur le jeu test")
    args = parser.parse_args()

    device = get_device()
    print(f"=== Palier 6 : Random Search NAS ({args.n_models} architectures, {args.epochs} époques, device: {device}) ===")

    train_loader, val_loader = cifar10_loaders(args.batch_size)
    test_loader = cifar10_test_loader() if args.test else None

    # 1. Échantillonnage de l'espace de recherche
    configs = generate_search_space(n_samples=args.n_models, seed=args.seed)
    print(f"{len(configs)} architectures distinctes échantillonnées avec succès.")

    # 2. Évaluation des architectures
    records: list[dict] = []
    for idx, cfg in enumerate(configs, 1):
        name = f"arch_{idx:02d}"
        print(f"[{idx:02d}/{args.n_models}] {name} : {summarize_config(cfg)} ...", flush=True)
        rec = evaluate_architecture(
            name=name,
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.train_seed,
        )
        print(
            f"      -> {rec['n_params']:,} params | {rec['macs_b1'] / 1e6:.2f} M MACs | "
            f"val_acc = {rec['val_acc'] * 100:.2f} % | train = {rec['train_seconds']:.1f} s"
        )
        append_result("s06", rec)
        records.append(rec)

    # 3. Calcul des Frontières de Pareto multi-objectifs
    obj_flops = [("val_acc", "max"), ("flops_b1", "min")]
    pareto_flops = {r["model_id"] for r in compute_pareto_front(records, obj_flops)}

    lat_key = "latency_mps_b64_ms" if device.type == "mps" else "latency_cpu_b64_ms"
    obj_latency = [("val_acc", "max"), (lat_key, "min")]
    pareto_latency = {r["model_id"] for r in compute_pareto_front(records, obj_latency)}

    obj_params = [("val_acc", "max"), ("n_params", "min")]
    pareto_params = {r["model_id"] for r in compute_pareto_front(records, obj_params)}

    # 4. Affichage du tableau récapitulatif
    print("\n" + "=" * 115)
    print(
        f"{'ID':6s} {'Architecture':38s} {'Params':>8s} {'MACs (M)':>9s} {'Val Acc':>8s} "
        f"{lat_key[8:-3]:>10s} {'Pareto (FLOPs/Lat/Par)':>24s}"
    )
    print("-" * 115)

    for r in sorted(records, key=lambda x: x["val_acc"], reverse=True):
        m_id = r["model_id"]
        f_mark = "FLOPs" if m_id in pareto_flops else "-"
        l_mark = "Lat" if m_id in pareto_latency else "-"
        p_mark = "Param" if m_id in pareto_params else "-"
        pareto_str = f"[{f_mark:^5s} | {l_mark:^3s} | {p_mark:^5s}]"

        print(
            f"{m_id:6s} {r['config_summary']:38s} {r['n_params']:8d} {r['macs_b1'] / 1e6:9.2f} "
            f"{r['val_acc'] * 100:7.2f}% {r[lat_key]:10.3f} ms {pareto_str:>24s}"
        )
    print("=" * 115)

    # 5. Synthèse comparative des frontières de Pareto
    print("\nFrontière de Pareto Précision vs FLOPs   :", sorted(pareto_flops))
    print(f"Frontière de Pareto Précision vs Latence ({lat_key}) :", sorted(pareto_latency))
    print("Frontière de Pareto Précision vs Paramètres :", sorted(pareto_params))

    divergence_flops_not_lat = pareto_flops - pareto_latency
    divergence_lat_not_flops = pareto_latency - pareto_flops

    if divergence_flops_not_lat:
        print(f"\nArchitecture(s) sur le front FLOPs mais DOMINÉE(S) en latence réelle : {sorted(divergence_flops_not_lat)}")
    if divergence_lat_not_flops:
        print(f"Architecture(s) sur le front Latence mais DOMINÉE(S) en FLOPs théoriques : {sorted(divergence_lat_not_flops)}")


if __name__ == "__main__":
    main()
