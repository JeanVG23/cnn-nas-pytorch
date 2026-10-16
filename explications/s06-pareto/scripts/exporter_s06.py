"""Exporte les vraies valeurs du palier 6 vers donnees/s06.js (chargé par la page, sans serveur).

Réutilise steps/s06_pareto.py sans le modifier. AUCUN entraînement et AUCUNE mesure ici : les 15 architectures, leurs précisions
et leurs latences viennent de experiments/results/s06.jsonl (palier déjà mesuré ; la prédiction de Jean est dans le rapport).
Jeu test : jamais lu (le JSONL n'a pas de test_acc et la page n'en parle pas).

Ce qui est exporté :
  * le code source cité par la page (lignes lues dans les vrais fichiers) ;
  * les 15 architectures (config, paramètres, MACs, val_acc, temps d'entraînement, 4 latences) ;
  * les fronts de Pareto calculés par le VRAI compute_pareto_front du palier, pour 7 coûts ;
  * la taille exacte de l'espace de recherche, comptée en rejouant le VRAI sample_config avec un « faux rng » qui énumère
    toutes les suites de choix possibles (aucune règle recopiée) ;
  * l'écart relatif entre 3 runs de latence et l'écart-type de la précision entre 3 graines, mesurés au palier 5 (experiments/results/s05.jsonl), pour calibrer la section « bruit ».

Usage (depuis la racine du projet) :
    uv run python explications/s06-pareto/scripts/exporter_s06.py
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from steps import s06_pareto  # noqa: E402
from steps.s06_pareto import compute_pareto_front, generate_search_space, sample_config  # noqa: E402

OUT_DIR = ROOT / "explications" / "s06-pareto" / "donnees"
SOURCES = ["steps/s06_pareto.py", "steps/s04_cnn_config.py"]
COSTS = ["n_params", "macs_b1", "flops_b1", "latency_mps_b64_ms", "latency_mps_b1_ms", "latency_cpu_b64_ms", "latency_cpu_b1_ms"]
LATENCY_KEYS = ["latency_mps_b1_ms", "latency_mps_b64_ms", "latency_cpu_b1_ms", "latency_cpu_b64_ms"]
KEEP = ["model_id", "config_summary", "config", "seed", "epochs", "n_params", "flops_b1", "macs_b1", "val_acc", "train_seconds", *LATENCY_KEYS, "device", "torch"]


class Done(Exception):
    """Plus de choix à rejouer dans ce chemin."""


class EnumRng:
    """Faux random.Random qui rejoue une suite de choix imposée, puis prend le premier choix et note le nombre d'options."""

    def __init__(self, script: list[int]) -> None:
        self.script, self.pos, self.arity = script, 0, []

    def _pick(self, n: int) -> int:
        self.arity.append(n)
        i = self.script[self.pos] if self.pos < len(self.script) else 0
        self.pos += 1
        return i

    def choice(self, seq):
        return seq[self._pick(len(seq))]

    def randint(self, a: int, b: int) -> int:
        return a + self._pick(b - a + 1)


def enumerate_space() -> dict:
    """Parcourt en profondeur toutes les suites de choix de sample_config ; renvoie les configs distinctes (par signature)."""
    seen: dict[tuple, int] = {}
    stack: list[list[int]] = [[]]
    runs = 0
    while stack:
        script = stack.pop()
        rng = EnumRng(script)
        cfg = sample_config(rng)
        runs += 1
        sig = tuple((s["channels"], s["kernel"], s["blocks"], s["pool"]) for s in cfg["stages"])
        seen[sig] = len(sig)
        # étendre : pour chaque choix pris au-delà du script, les autres options
        for k in range(len(script), len(rng.arity)):
            for alt in range(1, rng.arity[k]):
                stack.append([*script, *([0] * (k - len(script))), alt])
    by_stages = {}
    for n in seen.values():
        by_stages[n] = by_stages.get(n, 0) + 1
    return {"total": len(seen), "by_stages": by_stages, "paths": runs}


data: dict = {"sources": {f: (ROOT / f).read_text(encoding="utf-8").splitlines() for f in SOURCES}}

rows = [json.loads(line) for line in (ROOT / "experiments/results/s06.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(rows) == 15 and all("test_acc" not in r for r in rows)
assert {r["seed"] for r in rows} == {0} and {r["epochs"] for r in rows} == {3}
data["models"] = [{k: r[k] for k in KEEP} for r in rows]

# les 15 configs du JSONL sont bien celles que le palier tire avec la graine 42
assert [r["config"] for r in rows] == generate_search_space(15, seed=42), "le JSONL ne correspond plus à generate_search_space(15, 42)"

# fronts de Pareto : la vraie fonction du palier
data["fronts"] = {c: sorted(r["model_id"] for r in compute_pareto_front(rows, [("val_acc", "max"), (c, "min")])) for c in COSTS}

# espace de recherche : énumération exacte avec le vrai sample_config
space = enumerate_space()
data["space"] = space
data["constants"] = {"channels": list(s06_pareto.CHANNELS_POOL), "kernels": list(s06_pareto.KERNELS_POOL), "blocks": list(s06_pareto.BLOCKS_POOL)}

# bruit : écart relatif (max - min) / médiane entre les 3 runs finaux du palier 5, par config et par latence
s05 = [json.loads(line) for line in (ROOT / "experiments/results/s05.jsonl").read_text(encoding="utf-8").splitlines()]
final = [r for r in s05 if "test_acc" in r]
by_cfg: dict[str, list[dict]] = {}
for r in final:
    by_cfg.setdefault(r["config_name"], []).append(r)
noise = {}
for key in LATENCY_KEYS:
    vals = []
    for rs in by_cfg.values():
        v = [r[key] for r in rs]
        vals.append(round((max(v) - min(v)) / statistics.median(v), 4))
    noise[key] = vals
noise["val_std_pts"] = [round(statistics.stdev(r["val_acc"] for r in rs) * 100, 2) for rs in by_cfg.values()]  # écart-type de la précision entre 3 graines, en points
data["noise"] = noise

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "s06.js").write_text("window.S06 = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print("écrit", OUT_DIR / "s06.js", (OUT_DIR / "s06.js").stat().st_size // 1024, "Ko")
print("fronts", json.dumps(data["fronts"]))
print("espace", space)
print("bruit", {k: (min(v), statistics.median(v), max(v)) for k, v in noise.items()})
