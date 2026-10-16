"""Exporte les vraies valeurs PyTorch de l'étape 5 vers donnees/s05.js (chargé par la page, sans serveur).

Réutilise steps/s05_cost.py et steps/s04_cnn_config.py sans les modifier. AUCUN entraînement et AUCUNE mesure de latence ici :
les précisions et les latences viennent de experiments/results/s05.jsonl (le palier est déjà mesuré, la prédiction de Jean
est dans le rapport). Jeux lus : aucun (seulement des tenseurs aléatoires pour les formes et les FLOPs).
Le JSONL contient 24 lignes : 6 d'exploration (graine 0, sans test_acc) et 18 finales (3 graines x 6 configs, avec test_acc,
chiffres retenus du rapport). Seules les 18 lignes finales sont agrégées ; le jeu test (lu une fois pour le rapport) n'est PAS repris :
test_acc n'est jamais exporté, la page ne montre que la validation.

Ce qui est exporté :
  * le code source cité par la page (lignes lues dans les vrais fichiers) ;
  * pour les 6 configs de GRID : couches, formes, paramètres et MACs par couche, totaux (count_params, count_flops du palier) ;
  * des variantes de s03 (sans BatchNorm ni ReLU, avec biais, images 64x64) pour montrer ce que FlopCounterMode compte ;
  * une ligne par config, agrégée sur les 3 graines finales de experiments/results/s05.jsonl : val_acc = moyenne, val_std = écart-type
    (n - 1), latences = médiane des 3 runs (comme le rapport), lat_runs = les 3 valeurs brutes de chaque latence ;
  * des contrôles croisés (corrélations de rang, frontières de dominance) que la page recalcule en JavaScript.

Usage (depuis la racine du projet) :
    uv run python explications/s05-cost/scripts/exporter_s05.py
"""

import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from steps.s04_cnn_config import IN_CHANNELS, NUM_CLASSES, build_model  # noqa: E402
from steps.s05_cost import GRID, count_flops, count_params  # noqa: E402

OUT_DIR = ROOT / "explications" / "s05-cost" / "donnees"
SOURCES = ["steps/s05_cost.py", "steps/s04_cnn_config.py", "common/device.py", "tests/test_s05_cost.py"]
LATENCY_KEYS = ["latency_mps_b1_ms", "latency_mps_b64_ms", "latency_cpu_b1_ms", "latency_cpu_b64_ms"]


def module_macs(module: nn.Module, x: torch.Tensor) -> int:
    """MACs d'un seul module (FlopCounterMode, convolutions et Linear seulement) : FLOPs / 2."""
    with FlopCounterMode(display=False) as counter:  # sans no_grad : le suivi de modules du compteur exige un grad_fn sur la sortie d'un module isolé
        module(x.requires_grad_(True))
    return counter.get_total_flops() // 2


def trace(model) -> list[dict]:
    """Une ligne par module : étage, nom, forme de sortie, paramètres, MACs comptés, éléments de la sortie."""
    model.eval()
    x = torch.randn(1, IN_CHANNELS, 32, 32)
    rows = []

    def add(stage: int, module: nn.Module) -> torch.Tensor:
        nonlocal x
        macs = module_macs(module, x.detach())
        x = module(x).detach()
        rows.append({"stage": stage, "name": type(module).__name__, "out": list(x.shape[1:]), "params": sum(p.numel() for p in module.parameters()), "macs": macs, "elems": x[0].numel()})
        return x

    for s, stage_module in enumerate(model.stages, start=1):
        for module in stage_module:
            add(s, module)
    for module in model.head:
        add(0, module)
    return rows


def variant(config: dict, *, norm: bool = True, relu: bool = True, bias: bool = False) -> nn.Sequential:
    """Même réseau que ConfigCNN, avec ou sans BatchNorm / ReLU, avec ou sans biais (pour tester le compteur, sans toucher au palier)."""
    layers: list[nn.Module] = []
    c_in = IN_CHANNELS
    for st in config["stages"]:
        for b in range(st["blocks"]):
            layers.append(nn.Conv2d(c_in if b == 0 else st["channels"], st["channels"], st["kernel"], padding=st["kernel"] // 2, bias=bias))
            if norm:
                layers.append(nn.BatchNorm2d(st["channels"]))
            if relu:
                layers.append(nn.ReLU())
        if st["pool"]:
            layers.append(nn.MaxPool2d(2))
        c_in = st["channels"]
    layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(c_in, NUM_CLASSES)]
    return nn.Sequential(*layers)


data: dict = {"sources": {f: (ROOT / f).read_text(encoding="utf-8").splitlines() for f in SOURCES}}

# ---------------------------------------------------------------- les 6 configs de GRID
grid = {}
for name, config in GRID.items():
    model = build_model(config)
    layers = trace(model)
    flops = count_flops(model, (1, IN_CHANNELS, 32, 32))
    grid[name] = {"config": config, "layers": layers, "params": count_params(model), "flops": flops, "macs": flops // 2}
    assert sum(r["macs"] for r in layers) == flops // 2  # les MACs par couche somment au total du palier
data["grid"] = grid
assert grid["s03"]["params"] == 94_762 and grid["late_pool"]["params"] == 94_762  # même paramètres, FLOPs différents

# ---------------------------------------------------------------- ce que FlopCounterMode compte (variantes de s03)
s03 = GRID["s03"]
base = count_flops(variant(s03), (1, 3, 32, 32))
no_bn_relu = count_flops(variant(s03, norm=False, relu=False), (1, 3, 32, 32))
with_bias = count_flops(variant(s03, bias=True), (1, 3, 32, 32))
assert base == no_bn_relu == with_bias == grid["s03"]["flops"]  # BatchNorm, ReLU et biais ne changent rien au compte
m = build_model(s03)
at64 = count_flops(m, (1, 3, 64, 64))
data["counter"] = {
    "s03Flops": base,
    "withoutBnRelu": no_bn_relu,
    "withBias": with_bias,
    "paramsWithBias": count_params(variant(s03, bias=True)),
    "paramsNoBnRelu": count_params(variant(s03, norm=False, relu=False)),
    "size64": {"flops": at64, "macs": at64 // 2, "params": count_params(m)},  # mêmes paramètres, MACs presque x4 (la tête Linear ne change pas)
}

# ---------------------------------------------------------------- résultats du rapport : 3 graines finales agrégées par config
all_rows = [json.loads(line) for line in (ROOT / "experiments/results/s05.jsonl").read_text(encoding="utf-8").splitlines()]
final = [r for r in all_rows if "test_acc" in r]  # lignes finales (chiffres retenus) ; les lignes d'exploration (graine 0 seule) sont écartées
by_config: dict[str, list[dict]] = {}
for r in final:
    by_config.setdefault(r["config_name"], []).append(r)
assert sorted(by_config) == sorted(GRID), "il faut les lignes finales des 6 configs"
seeds = {tuple(sorted(r["seed"] for r in rs)) for rs in by_config.values()}
assert seeds == {(0, 1, 2)}, f"3 graines (0, 1, 2) attendues pour chaque config, trouvé {seeds}"
rows = []
for name, rs in by_config.items():  # ordre du JSONL : tiny, s03, wide, deep, kernel5, late_pool
    rs = sorted(rs, key=lambda r: r["seed"])
    for r in rs:
        assert r["n_params"] == grid[name]["params"] and r["flops_b1"] == grid[name]["flops"]  # le rapport et le recalcul concordent
    row = {k: rs[0][k] for k in ("config_name", "epochs", "n_params", "flops_b1", "macs_b1", "device", "torch")}
    row["n_seeds"] = len(rs)
    row["seeds"] = [r["seed"] for r in rs]
    row["val_acc"] = statistics.mean(r["val_acc"] for r in rs)
    row["val_std"] = statistics.stdev(r["val_acc"] for r in rs)  # écart-type d'échantillon (n - 1), comme le rapport
    row["val_runs"] = [r["val_acc"] for r in rs]
    row["train_seconds"] = statistics.mean(r["train_seconds"] for r in rs)
    for key in LATENCY_KEYS:
        row[key] = statistics.median(r[key] for r in rs)  # médiane des 3 runs, comme le rapport
    row["lat_runs"] = {key: [r[key] for r in rs] for key in LATENCY_KEYS}
    assert "test_acc" not in row  # le jeu test n'est pas repris par cette page
    rows.append(row)
data["runs"] = rows


# ---------------------------------------------------------------- contrôles croisés (recalculés par la page en JavaScript)
def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


names = [r["config_name"] for r in rows]
checks = {"spearman": {}, "inversions": {}, "front": {}}
for metric in ("n_params", "macs_b1"):
    for key in LATENCY_KEYS:
        mv, lv = [r[metric] for r in rows], [r[key] for r in rows]
        checks["spearman"][f"{metric}|{key}"] = round(spearman(mv, lv), 4)
        checks["inversions"][f"{metric}|{key}"] = sum(1 for i, j in itertools.combinations(range(len(rows)), 2) if mv[i] != mv[j] and (mv[i] - mv[j]) * (lv[i] - lv[j]) < 0)


def front(cost: str) -> list[str]:
    """Configs non dominées : aucune autre config n'a un coût <= et une précision >= avec une des deux inégalités stricte."""
    keep_names = []
    for r in rows:
        dominated = any(o is not r and o[cost] <= r[cost] and o["val_acc"] >= r["val_acc"] and (o[cost] < r[cost] or o["val_acc"] > r["val_acc"]) for o in rows)
        if not dominated:
            keep_names.append(r["config_name"])
    return sorted(keep_names)


for cost in ("n_params", "macs_b1", *LATENCY_KEYS):
    checks["front"][cost] = front(cost)
data["checks"] = checks

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "s05.js").write_text("window.S05 = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print("écrit", OUT_DIR / "s05.js", (OUT_DIR / "s05.js").stat().st_size // 1024, "Ko")
print({n: (v["params"], v["macs"]) for n, v in grid.items()})
print("compteur", data["counter"])
print("spearman", checks["spearman"])
print("inversions", checks["inversions"])
print("frontières", checks["front"])
