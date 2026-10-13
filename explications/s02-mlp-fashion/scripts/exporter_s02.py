"""Exporte les vraies valeurs PyTorch de l'étape 2 vers donnees/s02.js (chargé par la page, sans serveur).

Réutilise steps/s02_mlp_fashion.py et common/data.py sans les modifier. Aucun accès à FashionMNIST : le
palier 2 attend la prédiction de Jean avant tout entraînement, donc tout ce qui est exporté ici vient de
tenseurs synthétiques, de l'architecture, de la découpe d'indices et de petits modules PyTorch.
Usage (depuis la racine du projet) : uv run python explications/s02-mlp-fashion/scripts/exporter_s02.py
"""

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from common.data import (  # noqa: E402
    EVAL_BATCH,
    FASHION_MEAN,
    FASHION_STD,
    N_VAL,
    SPLIT_SEED,
    split_indices,
)
from common.device import set_seed  # noqa: E402
from steps.s02_mlp_fashion import MLP, evaluate, train_one_epoch  # noqa: E402

CPU = torch.device("cpu")
SOURCES = ["steps/s02_mlp_fashion.py", "common/data.py"]
N_TOTAL = (
    60_000  # FashionMNIST train : 60 000 images, dont N_VAL réservées à la validation
)
EPOCHS = 30
LRS = [1e-4, 1e-3, 1e-2, 1e-1]

data = {
    "sources": {f: (ROOT / f).read_text(encoding="utf-8").splitlines() for f in SOURCES}
}

# Architecture réelle : couches, formes, paramètres
model = MLP()
x = torch.randn(4, 1, 28, 28)
layers = []
for module in model.net:
    x = module(x)
    layers.append(
        {
            "name": type(module).__name__,
            "out": list(x.shape[1:]),
            "params": sum(p.numel() for p in module.parameters()),
            "shapes": [list(p.shape) for p in module.parameters()],
        }
    )
data["arch"] = {"layers": layers, "total": sum(p.numel() for p in model.parameters())}


def n_params(h1: int, h2: int) -> int:
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 10),
    )
    return sum(p.numel() for p in net.parameters())


data["paramsGrid"] = [
    {"h1": h1, "h2": h2, "total": n_params(h1, h2)}
    for h1, h2 in [
        (256, 128),
        (512, 128),
        (64, 32),
        (1024, 512),
        (128, 128),
        (10, 10),
        (784, 784),
    ]
]

# Normalisation et aplatissement
normalize = transforms.Normalize(FASHION_MEAN, FASHION_STD)
data["normalize"] = {
    "mean": FASHION_MEAN[0],
    "std": FASHION_STD[0],
    "inputs": [0.0, 0.5, 1.0],
    "outputs": normalize(torch.tensor([[[0.0, 0.5, 1.0]]]))[0, 0].tolist(),
}
pixels = torch.arange(784.0).reshape(1, 1, 28, 28)
flat = nn.Flatten()(pixels)
data["flatten"] = {
    "rowMajor": bool((flat[0] == torch.arange(784.0)).all()),
    "pixel_3_5": [28 * 3 + 5, flat[0, 28 * 3 + 5].item() == pixels[0, 0, 3, 5].item()],
    "shape": list(flat.shape),
}

# Deux Linear sans ReLU valent une seule Linear
torch.manual_seed(0)
l1, l2 = nn.Linear(5, 4).double(), nn.Linear(4, 3).double()
xs = torch.randn(6, 5).double()
w_comb, b_comb = l2.weight @ l1.weight, l2.weight @ l1.bias + l2.bias
single = xs @ w_comb.T + b_comb
data["stacked"] = {
    "diffLinearLinear": (l2(l1(xs)) - single).abs().max().item(),
    "diffWithRelu": (l2(torch.relu(l1(xs))) - single).abs().max().item(),
}

# Logits, softmax, CrossEntropyLoss : cas de référence
loss_fn = nn.CrossEntropyLoss()


def ce_case(logits_list: list[float], y: int) -> dict:
    logits = torch.tensor(logits_list, requires_grad=True)
    loss = loss_fn(logits[None], torch.tensor([y]))
    loss.backward()
    probs = torch.softmax(logits.detach(), dim=0)
    probs_in = probs.clone().requires_grad_(True)
    double = loss_fn(probs_in[None], torch.tensor([y]))
    double.backward()
    return {
        "logits": logits_list,
        "y": y,
        "probs": probs.tolist(),
        "loss": loss.item(),
        "grad": logits.grad.tolist(),
        "lossOnProbs": double.item(),
        "argmaxSame": bool(logits.argmax() == probs.argmax()),
    }


torch.manual_seed(3)
rand_logits = [round(v, 2) for v in torch.randn(10).tolist()]
data["ce"] = [
    ce_case([0.0] * 10, 3),
    ce_case([2.0] + [0.0] * 9, 0),
    ce_case([round(-4.5 + i, 2) for i in range(10)], 9),
    ce_case(rand_logits, 4),
]

# DataLoader : ordres réels sur un mini jeu de 20 éléments
toy = TensorDataset(torch.arange(20))


def epochs_order(
    seed: int, batch_size: int, shuffle: bool, n_epochs: int = 3
) -> list[list[int]]:
    torch.manual_seed(seed)
    loader = DataLoader(toy, batch_size=batch_size, shuffle=shuffle)
    return [[int(i) for (b,) in loader for i in b] for _ in range(n_epochs)]


data["loader"] = {
    "n": 20,
    "shuffled": {str(seed): epochs_order(seed, 4, True) for seed in range(5)},
    "sequential": epochs_order(0, 4, False),
    "orderIndependentOfBatchSize": all(
        epochs_order(0, b, True) == epochs_order(0, 4, True) for b in (2, 5, 10)
    ),
}


def batch_count(n: int, batch_size: int) -> dict:
    loader = DataLoader(list(range(n)), batch_size=batch_size)
    return {"batches": len(loader), "last": n - (len(loader) - 1) * batch_size}


n_train = N_TOTAL - N_VAL
data["batchCounts"] = {
    "nTrain": n_train,
    "nVal": N_VAL,
    "evalBatch": EVAL_BATCH,
    "train": {str(b): batch_count(n_train, b) for b in [16, 32, 64, 128, 256, 512]},
    "val": batch_count(N_VAL, EVAL_BATCH),
}

# Découpe train/val déterministe
train_idx, val_idx = split_indices(N_TOTAL, N_VAL)
heads_by_model_seed = []
for model_seed in (0, 1, 2):
    set_seed(model_seed)
    heads_by_model_seed.append(split_indices(N_TOTAL, N_VAL)[1][:8])
data["split"] = {
    "splitSeed": SPLIT_SEED,
    "nTrain": len(train_idx),
    "nVal": len(val_idx),
    "valHead": val_idx[:8],
    "trainHead": train_idx[:8],
    "disjoint": set(train_idx).isdisjoint(val_idx),
    "coversAll": sorted(train_idx + val_idx) == list(range(N_TOTAL)),
    "valHeadOtherSplitSeed": split_indices(N_TOTAL, N_VAL, split_seed=SPLIT_SEED + 1)[
        1
    ][:8],
    "sameValForModelSeeds": all(
        h == heads_by_model_seed[0] for h in heads_by_model_seed
    ),
}


# Entraînement sur des tenseurs synthétiques (jamais sur FashionMNIST)
def make_set(seed: int, n: int, rule: bool) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 1, 28, 28, generator=generator)
    labels = (
        images[:, 0, 0, :10].argmax(dim=1)
        if rule
        else torch.randint(0, 10, (n,), generator=generator)
    )
    return images, labels


def train_one_epoch_without_zero_grad(model, loader, loss_fn, optimizer) -> float:
    """Copie de train_one_epoch SANS optimizer.zero_grad() : expérience du script d'export, pas du palier."""
    model.train()
    total, n = 0.0, 0
    for x_batch, y_batch in loader:
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y_batch)
        n += len(y_batch)
    return total / n


def run(train_set, held_set, lr: float, zero_grad: bool) -> dict:
    train_loader = DataLoader(TensorDataset(*train_set), batch_size=64, shuffle=True)
    eval_train = DataLoader(TensorDataset(*train_set), batch_size=EVAL_BATCH)
    eval_held = DataLoader(TensorDataset(*held_set), batch_size=EVAL_BATCH)
    set_seed(0)
    mlp = MLP()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    out = {
        "trainLoss": [],
        "evalTrainLoss": [],
        "evalTrainAcc": [],
        "heldLoss": [],
        "heldAcc": [],
    }

    def snapshot() -> None:
        loss_t, acc_t = evaluate(mlp, eval_train, loss_fn, CPU)
        loss_h, acc_h = evaluate(mlp, eval_held, loss_fn, CPU)
        out["evalTrainLoss"].append(loss_t)
        out["evalTrainAcc"].append(acc_t)
        out["heldLoss"].append(loss_h)
        out["heldAcc"].append(acc_h)

    snapshot()
    for _ in range(EPOCHS):
        if zero_grad:
            out["trainLoss"].append(
                train_one_epoch(mlp, train_loader, loss_fn, optimizer, CPU)
            )
        else:
            out["trainLoss"].append(
                train_one_epoch_without_zero_grad(mlp, train_loader, loss_fn, optimizer)
            )
        snapshot()
    return out


N_SYNTH, N_HELD = 2048, 512
synthetic = {}
for name, rule in [("random", False), ("rule", True)]:
    train_set, held_set = make_set(0, N_SYNTH, rule), make_set(1, N_HELD, rule)
    runs = {str(lr): run(train_set, held_set, lr, zero_grad=True) for lr in LRS}
    runs["0.001-nozero"] = run(train_set, held_set, 1e-3, zero_grad=False)
    synthetic[name] = {"nTrain": N_SYNTH, "nHeld": N_HELD, "runs": runs}
data["train"] = {"epochs": EPOCHS, "lrs": LRS, "batchSize": 64, "datasets": synthetic}

# train() / eval() / no_grad
set_seed(0)
mlp = MLP()
batch = torch.randn(8, 1, 28, 28)
mlp.train()
out_train = mlp(batch)
mlp.eval()
out_eval = mlp(batch)
with torch.no_grad():
    out_no_grad = mlp(batch)
ones = torch.ones(10)
dropout_runs = {}
for p in (0.25, 0.5):
    draws = []
    for seed in range(3):
        torch.manual_seed(seed)
        drop = nn.Dropout(p)
        drop.train()
        draws.append(drop(ones).tolist())
    dropout_runs[str(p)] = {"train": draws, "eval": nn.Dropout(p).eval()(ones).tolist()}
bn = nn.BatchNorm1d(1)
sample = torch.tensor([[1.0], [2.0], [3.0], [10.0]])
bn.train()
bn_train = bn(sample).flatten().tolist()
running = {"mean": bn.running_mean.item(), "var": bn.running_var.item()}
bn.eval()
bn_eval = bn(sample).flatten().tolist()
data["modes"] = {
    "mlpMaxDiffTrainEval": (out_train - out_eval).abs().max().item(),
    "requiresGradTrain": out_train.requires_grad,
    "requiresGradEval": out_eval.requires_grad,
    "requiresGradNoGrad": out_no_grad.requires_grad,
    "hasDropoutOrBatchNorm": any(
        isinstance(m, (nn.Dropout, nn.BatchNorm1d)) for m in mlp.modules()
    ),
    "dropout": dropout_runs,
    "batchNorm": {
        "input": [v[0] for v in sample.tolist()],
        "train": bn_train,
        "eval": bn_eval,
        "running": running,
    },
}

# Le vrai MLP à l'initialisation (set_seed(0), non entraîné) : poids exportés pour que la page fasse la vraie passe avant


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(
        t.detach().contiguous().numpy().astype("<f4").tobytes()
    ).decode("ascii")


set_seed(0)
init = MLP()
linear = [m for m in init.net if isinstance(m, nn.Linear)]
ref_input = torch.randn(1, 1, 28, 28, generator=torch.Generator().manual_seed(5))
with torch.no_grad():
    ref_logits = init(ref_input)[0].tolist()
data["classes"] = list(datasets.FashionMNIST.classes)
data["init"] = {
    "seed": 0,
    "weights": [
        {"w": b64(m.weight), "b": b64(m.bias), "shape": list(m.weight.shape)}
        for m in linear
    ],
    "refInput": b64(ref_input),
    "refLogits": ref_logits,
}

data["env"] = {"torch": torch.__version__, "device": "cpu"}

out_path = Path(__file__).resolve().parents[1] / "donnees" / "s02.js"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(
    "window.S02 = " + json.dumps(data, indent=1, ensure_ascii=False) + ";\n",
    encoding="utf-8",
)
print(f"écrit : {out_path}")
