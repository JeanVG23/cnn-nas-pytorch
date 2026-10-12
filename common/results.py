"""Résultats bruts : une ligne JSON par exécution, dans experiments/results/<palier>.jsonl.

Fournit aussi une synthèse groupée par paramètre (moyenne, écart-type sur 3 graines).
Usage de la synthèse : uv run python -m common.results s03 epochs val_acc
"""

import json
import statistics
import sys
from pathlib import Path

RESULTS_DIR = Path("experiments/results")


def append_result(step: str, record: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Ajoute une ligne JSON dans <results_dir>/<step>.jsonl et renvoie le chemin."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{step}.jsonl"
    payload = {"step": step, **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def summarize(step: str, group_by: str, metric: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Groupe les exécutions par group_by et calcule n, mean, std de metric."""
    groups: dict[object, list[float]] = {}
    path = results_dir / f"{step}.jsonl"
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        groups.setdefault(record[group_by], []).append(record[metric])
    return [
        {group_by: key, "n": len(values), "mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None}
        for key, values in sorted(groups.items(), key=lambda item: str(item[0]))
    ]


if __name__ == "__main__":
    for row in summarize(*sys.argv[1:4]):
        print(row)
