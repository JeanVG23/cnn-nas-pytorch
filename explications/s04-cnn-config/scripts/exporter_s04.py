"""Exporte les vraies valeurs PyTorch de l'étape 4 vers donnees/s04.js (chargé par la page, sans serveur).

Réutilise steps/s04_cnn_config.py sans le modifier. AUCUN entraînement ici : les résultats (précision, temps)
viennent de experiments/results/s04.jsonl (le palier est déjà mesuré, la prédiction de Jean est dans le rapport).
Jeux lus : aucun (seulement des tenseurs aléatoires pour les formes, les MACs et le champ réceptif).

Ce qui est exporté :
  * le code source cité par la page (lignes lues dans les vrais fichiers) ;
  * pour les 3 configs du palier : couches, formes, paramètres par couche, MACs (FlopCounterMode, 1 MAC = 2 FLOPs) ;
  * les messages d'erreur réels de validate_config sur des configs invalides ;
  * l'effet d'un noyau pair avec padding = kernel // 2 (taille de sortie réelle) ;
  * le piège « liste Python au lieu de nn.ModuleList » (paramètres, state_dict, .eval(), .double()) ;
  * une grille largeur x blocs (paramètres et MACs, construits par PyTorch) ;
  * le champ réceptif mesuré par gradient sur une variante (AvgPool, sans ReLU) de chaque config ;
  * les 3 lignes de experiments/results/s04.jsonl, et les temps / précisions du palier 3 (même architecture que
    s03_equivalent, 15 époques) pour comparer : experiments/results/s03.jsonl (sans le jeu test).

Usage (depuis la racine du projet) :
    uv run python explications/s04-cnn-config/scripts/exporter_s04.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from steps.s04_cnn_config import (  # noqa: E402
    CONFIGS,
    IN_CHANNELS,
    NUM_CLASSES,
    build_model,
    conv_stage,
    stage,
)

OUT_DIR = ROOT / "explications" / "s04-cnn-config" / "donnees"
SOURCES = ["steps/s04_cnn_config.py", "common/train.py"]


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def macs(model: nn.Module) -> int:
    """MACs de la passe avant sur une image 32x32 (convolutions et Linear seulement) : FLOPs / 2."""
    model.eval()
    with FlopCounterMode(display=False) as counter:
        model(torch.randn(1, IN_CHANNELS, 32, 32))
    return counter.get_total_flops() // 2


def trace(model) -> list[dict]:
    """Une ligne par module : étage, nom, forme de sortie (C, H, W) ou (N,), paramètres."""
    model.eval()
    x = torch.randn(1, IN_CHANNELS, 32, 32)
    rows = []
    for s, stage_module in enumerate(model.stages, start=1):
        for module in stage_module:
            x = module(x)
            rows.append({"stage": s, "name": type(module).__name__, "out": list(x.shape[1:]), "params": n_params(module)})
    for module in model.head:
        x = module(x)
        rows.append({"stage": 0, "name": type(module).__name__, "out": list(x.shape[1:]), "params": n_params(module)})
    return rows


data: dict = {"sources": {f: (ROOT / f).read_text(encoding="utf-8").splitlines() for f in SOURCES}}

# ---------------------------------------------------------------- les 3 configs du palier
named = {}
for name, config in CONFIGS.items():
    model = build_model(config)
    named[name] = {"config": config, "layers": trace(model), "total": n_params(model), "macs": macs(model)}
assert named["s03_equivalent"]["total"] == 94_762 and named["wide"]["total"] == 373_834 and named["deep"]["total"] == 288_746
data["named"] = named

# ---------------------------------------------------------------- configs invalides : messages réels
invalid_cases = {
    "vide": {},
    "étage sans clé pool": {"stages": [{"channels": 8, "kernel": 3, "blocks": 1}]},
    "noyau pair (4)": {"stages": [stage(8, kernel=4)]},
    "0 canal": {"stages": [stage(0)]},
    "blocks booléen": {"stages": [{"channels": 8, "kernel": 3, "blocks": True, "pool": True}]},
    "pool « oui »": {"stages": [{"channels": 8, "kernel": 3, "blocks": 1, "pool": "oui"}]},
    "5 étages avec pooling": {"stages": [stage(4) for _ in range(5)]},
    "6 étages avec pooling": {"stages": [stage(4) for _ in range(6)]},
}
invalid = []
for label, config in invalid_cases.items():
    try:
        build_model(config)
        invalid.append({"label": label, "config": config, "error": None})
    except ValueError as e:
        invalid.append({"label": label, "config": config, "error": str(e)})
data["invalid"] = invalid
data["fiveStages"] = {"out": list(build_model(invalid_cases["5 étages avec pooling"]).eval()(torch.randn(1, 3, 32, 32)).shape)}

# Noyau pair : pourquoi validate_config le refuse (padding = kernel // 2 ne conserve plus la taille)
data["kernelSizes"] = [
    {"k": k, "padding": k // 2, "out": nn.Conv2d(3, 8, k, padding=k // 2)(torch.randn(1, 3, 32, 32)).shape[-1]} for k in [1, 2, 3, 4, 5, 6, 7]
]

# ---------------------------------------------------------------- piège : liste Python au lieu de nn.ModuleList
class ListCNN(nn.Module):
    """Même réseau que ConfigCNN, mais les étages sont rangés dans une liste Python (le piège que le code évite)."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.stages = []
        c_in = IN_CHANNELS
        for st in config["stages"]:
            self.stages.append(conv_stage(c_in, st))
            c_in = st["channels"]
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(c_in, NUM_CLASSES))

    def forward(self, x):
        for m in self.stages:
            x = m(x)
        return self.head(x)


trap = {}
for name, config in CONFIGS.items():
    good, bad = build_model(config), ListCNN(config)
    good.eval()
    bad.eval()
    forward_runs = list(bad(torch.zeros(1, 3, 32, 32)).shape) == [1, 10]  # le réseau à liste Python calcule bien, avant .double()
    good.double()
    bad.double()
    trap[name] = {
        "forwardRuns": forward_runs,
        "stageStillTrainingAfterEvalModuleList": any(m.training for st_ in good.stages for m in st_.modules()),
        "stageDtypeAfterDoubleModuleList": str(good.stages[0][0].weight.dtype).replace("torch.", ""),
        "paramsModuleList": n_params(good),
        "paramsList": n_params(bad),
        "stateKeysModuleList": len(good.state_dict()),
        "stateKeysList": len(bad.state_dict()),
        "stageStillTrainingAfterEval": any(m.training for st_ in bad.stages for m in st_.modules()),
        "stageDtypeAfterDouble": str(bad.stages[0][0].weight.dtype).replace("torch.", ""),
        "headDtypeAfterDouble": str(bad.head[2].weight.dtype).replace("torch.", ""),
    }
data["listTrap"] = trap

# ---------------------------------------------------------------- grille largeur x blocs (construite par PyTorch)
BASE = [32, 64, 128]
WIDTHS = [0.5, 1, 1.5, 2, 3, 4]
BLOCKS = [1, 2, 3, 4]
grid = []
for m in WIDTHS:
    for b in BLOCKS:
        cfg = {"stages": [stage(int(c * m), blocks=b) for c in BASE]}
        model = build_model(cfg)
        grid.append({"mult": m, "blocks": b, "channels": [int(c * m) for c in BASE], "total": n_params(model), "macs": macs(model)})
data["grid"] = {"base": BASE, "widths": WIDTHS, "blocks": BLOCKS, "cells": grid}
iso = build_model({"stages": [stage(c) for c in (56, 112, 224)]})
data["iso"] = {"channels": [56, 112, 224], "total": n_params(iso), "macs": macs(iso)}


# ---------------------------------------------------------------- champ réceptif mesuré par gradient
def receptive_field(config: dict) -> int:
    """Étendue (en pixels d'entrée) du gradient d'un pixel central de la sortie du dernier étage.

    Variante du réseau pour la mesure : MaxPool remplacé par AvgPool (même champ réceptif, gradient non nul partout),
    ni BatchNorm ni ReLU (qui annulent des gradients sans changer le champ réceptif théorique).
    """
    layers: list[nn.Module] = []
    c_in = IN_CHANNELS
    for st in config["stages"]:
        for b in range(st["blocks"]):
            layers.append(nn.Conv2d(c_in if b == 0 else st["channels"], st["channels"], st["kernel"], padding=st["kernel"] // 2, bias=False))
        if st["pool"]:
            layers.append(nn.AvgPool2d(2))
        c_in = st["channels"]
    net = nn.Sequential(*layers)
    size = 256
    x = torch.randn(1, IN_CHANNELS, size, size, requires_grad=True)
    y = net(x)
    y[0, 0, y.shape[2] // 2, y.shape[3] // 2].backward()
    rows = (x.grad.abs().sum(dim=(0, 1)) > 0).any(dim=1).nonzero().flatten()
    return int(rows.max() - rows.min() + 1)


RF_EXTRA = {
    "noyau 5, puis noyau 7 en 2 blocs sans pooling, puis noyau 3": {"stages": [stage(8, kernel=5), stage(8, kernel=7, blocks=2, pool=False), stage(16)]},
    "un seul étage, 1 conv 3x3, sans pooling": {"stages": [stage(4, pool=False)]},
}
data["receptive"] = [{"name": n, "config": c, "rf": receptive_field(c)} for n, c in CONFIGS.items()] + [
    {"name": n, "config": c, "rf": receptive_field(c)} for n, c in RF_EXTRA.items()
]

# ---------------------------------------------------------------- résultats bruts du rapport
rows = [json.loads(line) for line in (ROOT / "experiments/results/s04.jsonl").read_text(encoding="utf-8").splitlines()]
data["runs"] = [{k: v for k, v in row.items() if k in ("config_name", "seed", "epochs", "n_params", "train_seconds", "val_acc", "val_loss", "device", "torch")} for row in rows]
s03_rows = [json.loads(line) for line in (ROOT / "experiments/results/s03.jsonl").read_text(encoding="utf-8").splitlines()]
data["s03Ref"] = [
    {"seed": r["seed"], "epochs": r["epochs"], "train_seconds": r["train_seconds"], "epoch_mean": round(r["train_seconds"] / r["epochs"], 2), "val_acc": r["val_acc"], "n_params": r["n_params"]}
    for r in s03_rows
]
assert {r["config_name"]: r["n_params"] for r in data["runs"]} == {n: v["total"] for n, v in named.items()}

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "s04.js").write_text("window.S04 = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print("écrit", OUT_DIR / "s04.js", (OUT_DIR / "s04.js").stat().st_size // 1024, "Ko")
print({n: (v["total"], v["macs"]) for n, v in named.items()})
print("iso", data["iso"])
print("rf", [(r["name"][:12], r["rf"]) for r in data["receptive"]])
print("piège", trap["deep"])
print("noyaux", data["kernelSizes"])
print("erreurs", [(i["label"], i["error"]) for i in invalid])
