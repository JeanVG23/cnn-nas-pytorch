"""Exporte les vraies valeurs PyTorch de l'étape 3 vers donnees/s03.js (chargé par la page, sans serveur).

Réutilise steps/s03_cnn_cifar.py, common/data.py et common/train.py sans les modifier.
Jeux lus : entraînement et validation de CIFAR-10 uniquement (images affichées, précision du checkpoint,
courbe d'entraînement). Le jeu test n'est JAMAIS chargé ici : ses chiffres viennent du rapport
experiments/s03-cnn-cifar.md et de experiments/results/s03.jsonl.

Ce qui est exporté :
  * le code source cité par la page (lignes lues dans les vrais fichiers) ;
  * l'architecture et les formes (CifarCNN réel), le nombre de paramètres de plusieurs configs ;
  * les poids du checkpoint data/checkpoints/s03_cnn_cifar.pt (dernier run, graine 2) : la page recalcule
    le passage avant en JavaScript et compare avec les sorties PyTorch exportées ici ;
  * 10 images de validation (une par classe) ;
  * le planning OneCycleLR réel et, avec --retrain, la courbe d'un entraînement de graine 0 relancé
    (validation seulement, environ 2 minutes sur mps), mise en cache dans donnees/historique_s03.json.

Usage (depuis la racine du projet) :
    uv run python explications/s03-cnn-cifar/scripts/exporter_s03.py [--retrain]
"""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import datasets  # noqa: E402

from common.data import (  # noqa: E402
    CIFAR_MEAN,
    CIFAR_STD,
    N_VAL,
    cifar10_loaders,
    split_indices,
)
from common.device import get_device, set_seed  # noqa: E402
from common.train import evaluate, fit  # noqa: E402
from steps.s03_cnn_cifar import CHECKPOINT, CifarCNN, load_checkpoint, trace_shapes  # noqa: E402

CPU = torch.device("cpu")
OUT_DIR = ROOT / "explications" / "s03-cnn-cifar" / "donnees"
HISTORY_CACHE = OUT_DIR / "historique_s03.json"
SOURCES = ["steps/s03_cnn_cifar.py", "common/train.py", "common/data.py"]
CLASSES = ["avion", "automobile", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "bateau", "camion"]
EPOCHS, LR, BATCH = 15, 3e-3, 128


def r(values, digits=6):
    """Liste de flottants arrondie à `digits` chiffres significatifs (le JS reste léger)."""
    return [float(f"{v:.{digits}g}") for v in values]


def flat(t: torch.Tensor, digits=6) -> list[float]:
    return r(t.detach().reshape(-1).tolist(), digits)


parser = argparse.ArgumentParser()
parser.add_argument("--retrain", action="store_true", help="relancer l'entraînement de graine 0 (val seulement)")
args = parser.parse_args()
OUT_DIR.mkdir(parents=True, exist_ok=True)

data: dict = {"sources": {f: (ROOT / f).read_text(encoding="utf-8").splitlines() for f in SOURCES}}
data["classes"] = CLASSES
data["normalize"] = {"mean": list(CIFAR_MEAN), "std": list(CIFAR_STD)}

# ---------------------------------------------------------------- architecture et formes (CifarCNN réel)
model = CifarCNN().eval()
x = torch.randn(1, 3, 32, 32)
layers = []
for b, block in enumerate(model.features, start=1):
    for module in block:
        x = module(x)
        layers.append({"block": b, "name": type(module).__name__, "out": list(x.shape[1:]), "params": sum(p.numel() for p in module.parameters())})
for module in model.head:
    x = module(x)
    layers.append({"block": 4, "name": type(module).__name__, "out": list(x.shape[1:]), "params": sum(p.numel() for p in module.parameters())})
data["arch"] = {"layers": layers, "total": sum(p.numel() for p in model.parameters())}
sd = model.state_dict()
data["stateDict"] = {
    "keys": len(sd),
    "elements": sum(v.numel() for v in sd.values()),
    "buffers": [k for k in sd if "running" in k or "num_batches" in k][:3],
}

shape_runs = {}
for size in [32, 30, 28, 16, 64, 8, 7, 4]:
    try:
        shape_runs[str(size)] = {"ok": True, "shapes": [[n, list(s)] for n, s in trace_shapes(model, torch.randn(1, 3, size, size))]}
    except RuntimeError as e:
        shape_runs[str(size)] = {"ok": False, "error": str(e)}
data["shapeRuns"] = shape_runs


# ---------------------------------------------------------------- nombre de paramètres de configs construites avec PyTorch
def build(c1: int, c2: int, c3: int, k: int, bias: bool) -> nn.Sequential:
    def blk(a: int, b: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(a, b, kernel_size=k, padding=k // 2, bias=bias), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(blk(3, c1), blk(c1, c2), blk(c2, c3), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(c3, 10))


def count(c1, c2, c3, k, bias) -> int:
    return sum(p.numel() for p in build(c1, c2, c3, k, bias).parameters())


GRID = [(32, 64, 128, 3, False), (64, 128, 256, 3, False), (16, 32, 64, 3, False), (32, 64, 128, 5, False), (32, 64, 128, 3, True),
        (32, 64, 128, 1, False), (8, 8, 8, 3, False), (128, 128, 128, 3, False), (24, 48, 96, 3, False), (32, 64, 96, 3, False)]
data["paramsGrid"] = [{"c": [a, b, c], "k": k, "bias": bias, "total": count(a, b, c, k, bias)} for a, b, c, k, bias in GRID]
assert data["paramsGrid"][0]["total"] == data["arch"]["total"] == 94_762
# Une solution proche de la prédiction de Jean (60 000), cherchée par force brute sur des canaux multiples de 4
best = min(((a, b, c) for a in range(8, 65, 4) for b in range(a, 129, 4) for c in range(b, 257, 4)), key=lambda t: abs(count(*t, 3, False) - 60_000))
data["mission"] = {"target": 60_000, "solution": list(best), "total": count(*best, 3, False)}

# ---------------------------------------------------------------- checkpoint : poids, images de validation, vérifications
reloaded = CifarCNN()
load_checkpoint(reloaded, CHECKPOINT)
reloaded.eval()
weights = []
for blk in reloaded.features:
    conv, bn = blk[0], blk[1]
    weights.append({
        "cin": conv.in_channels, "cout": conv.out_channels,
        "w": flat(conv.weight), "gamma": flat(bn.weight), "beta": flat(bn.bias),
        "mean": flat(bn.running_mean), "var": flat(bn.running_var), "eps": bn.eps, "momentum": bn.momentum,
    })
lin = reloaded.head[2]
data["weights"] = {"blocks": weights, "linW": flat(lin.weight), "linB": flat(lin.bias)}
data["checkpoint"] = {"file": str(CHECKPOINT), "bytes": (ROOT / CHECKPOINT).stat().st_size}

raw = datasets.CIFAR10("data", train=True, download=True)  # images brutes (uint8), pas de transformation
_, val_idx = split_indices(len(raw), N_VAL)
picked = {}
for i in val_idx:
    picked.setdefault(raw.targets[i], i)
    if len(picked) == 10:
        break
images = []
mean_t, std_t = torch.tensor(CIFAR_MEAN).view(3, 1, 1), torch.tensor(CIFAR_STD).view(3, 1, 1)
batch = []
for label in range(10):
    arr = torch.tensor(list(raw[picked[label]][0].getdata()), dtype=torch.float32).view(32, 32, 3).permute(2, 0, 1)  # (3, 32, 32)
    batch.append(((arr / 255 - mean_t) / std_t))
    images.append({"label": label, "valIndex": picked[label], "pixels": [int(v) for v in arr.reshape(-1).tolist()]})
xb = torch.stack(batch)  # (10, 3, 32, 32)
with torch.no_grad():
    logits = reloaded(xb)
    feats = reloaded.head[0](reloaded.features(xb)).flatten(1)  # (10, 128)
    conv1_out = reloaded.features[0][0](xb)  # (10, 32, 32, 32), avant BatchNorm
for k, im in enumerate(images):
    im["logits"] = r(logits[k].tolist(), 5)
    im["pooled"] = r(feats[k].tolist(), 5)
data["images"] = images
data["conv1Check"] = {  # image 0 : sortie de la première convolution (avant BN) en quelques positions
    "positions": [[0, 0], [5, 7], [16, 16], [31, 31]],
    "values": [[r(conv1_out[0, :, y, x].tolist(), 6) for y, x in [(0, 0), (5, 7), (16, 16), (31, 31)]]][0],
}

val_loader = cifar10_loaders(BATCH)[1]
val_loss, val_acc = evaluate(reloaded, val_loader, CPU)
data["checkpointVal"] = {"val_loss": val_loss, "val_acc": val_acc, "n": N_VAL}

# ---------------------------------------------------------------- BatchNorm : vérifications faites par PyTorch
bn_train = nn.BatchNorm2d(32).train()
with torch.no_grad():
    base = bn_train(conv1_out)
    shifted = bn_train(conv1_out + 5.0)
c0 = conv1_out[:, 0]  # canal 0 sur tout le lot : (10, 32, 32)
fresh = nn.BatchNorm2d(32).train()
with torch.no_grad():
    fresh(conv1_out)
polluted = copy.deepcopy(reloaded).train()
with torch.no_grad():
    polluted(xb[:2])  # le « passage à vide » de la ligne 86 du palier, mais en mode train
polluted.eval()
loss_p, acc_p = evaluate(polluted, val_loader, CPU)
gamma_beta_diff = max(
    max((polluted.features[i][1].weight - reloaded.features[i][1].weight).abs().max().item(), (polluted.features[i][1].bias - reloaded.features[i][1].bias).abs().max().item())
    for i in range(3)
)
data["bn"] = {
    "batchMean0": c0.mean().item(), "batchVar0": c0.var(unbiased=False).item(), "batchVarUnbiased0": c0.var(unbiased=True).item(),
    "shiftMaxDiff": (shifted - base).abs().max().item(),
    "freshRunningMean0": fresh.running_mean[0].item(), "freshRunningVar0": fresh.running_var[0].item(),
    "dryPass": {
        "runningMean0Before": reloaded.features[0][1].running_mean[0].item(), "runningMean0After": polluted.features[0][1].running_mean[0].item(),
        "valAccBefore": val_acc, "valAccAfter": acc_p, "valLossAfter": loss_p, "gammaBetaMaxDiff": gamma_beta_diff,
    },
}

# ---------------------------------------------------------------- OneCycleLR réel (lr de pic 3e-3, 45 000 images, batch 128)
n_batches = len(cifar10_loaders(BATCH)[0])
opt = torch.optim.AdamW(nn.Linear(1, 1).parameters(), lr=LR, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * n_batches)
lrs, betas = [], []
for _ in range(EPOCHS * n_batches):
    lrs.append(opt.param_groups[0]["lr"])
    betas.append(opt.param_groups[0]["betas"][0])
    opt.step()
    sched.step()
data["schedule"] = {"batchesPerEpoch": n_batches, "epochs": EPOCHS, "lr": r(lrs, 4), "beta1": r(betas, 4), "peakStep": max(range(len(lrs)), key=lrs.__getitem__)}

# ---------------------------------------------------------------- courbe d'entraînement (graine 0, validation seulement), mise en cache
if args.retrain or not HISTORY_CACHE.exists():
    device = get_device()
    set_seed(0)
    train_loader, vloader = cifar10_loaders(BATCH)
    net = CifarCNN().to(device)
    next(iter(vloader))  # même séquence d'appels que le palier (consomme le générateur comme lui)
    result = fit(net, train_loader, vloader, epochs=EPOCHS, lr=LR, device=device)
    HISTORY_CACHE.write_text(json.dumps({"seed": 0, "device": device.type, "torch": torch.__version__, "train_seconds": result["train_seconds"], "history": result["history"]}))
data["run"] = json.loads(HISTORY_CACHE.read_text())

# ---------------------------------------------------------------- résultats bruts du rapport (jsonl du palier)
rows = [json.loads(line) for line in (ROOT / "experiments/results/s03.jsonl").read_text(encoding="utf-8").splitlines()]
data["runs"] = [{k: v for k, v in row.items() if k in ("seed", "val_acc", "val_loss", "test_acc", "test_loss", "train_seconds", "epoch_seconds", "n_params")} for row in rows]

(OUT_DIR / "s03.js").write_text("window.S03 = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print("écrit", OUT_DIR / "s03.js", (OUT_DIR / "s03.js").stat().st_size // 1024, "Ko")
print("val checkpoint", val_loss, val_acc, "| après passage à vide en train :", loss_p, acc_p)
print("mission", data["mission"], "| run :", data["run"]["history"][-1])
