"""Tests hermétiques et rapides pour le palier 6 (Random Search et Frontière de Pareto).

Exécutés sur CPU, sans réseau, avec données synthétiques.
"""

import random

import pytest
import torch

from steps.s04_cnn_config import build_model, validate_config
from steps.s06_pareto import (
    compute_pareto_front,
    generate_search_space,
    is_dominated,
    sample_config,
)


def test_sample_config_produces_valid_configs():
    """Vérifie que 50 tirages aléatoires sont tous conformes aux règles de validate_config."""
    rng = random.Random(1234)
    for _ in range(50):
        cfg = sample_config(rng)
        validate_config(cfg)  # ne doit pas lever d'exception


def test_generate_search_space_is_deterministic_and_unique():
    """La génération d'espace avec la même graine produit exactement les mêmes configs sans doublons."""
    space1 = generate_search_space(n_samples=15, seed=42)
    space2 = generate_search_space(n_samples=15, seed=42)
    assert len(space1) == 15
    assert space1 == space2

    # Vérification d'unicité
    representations = [repr(cfg) for cfg in space1]
    assert len(set(representations)) == len(space1)


def test_every_sampled_model_can_forward_synthetic_batch():
    """Vérifie que les architectures échantillonnées s'instancient et exécutent une passe avant sur CPU."""
    space = generate_search_space(n_samples=5, seed=99)
    x = torch.randn(2, 3, 32, 32)
    for cfg in space:
        model = build_model(cfg).eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)


def test_pareto_single_point_is_optimal():
    """Un ensemble à un seul point est trivialement sur le front de Pareto."""
    single = [{"name": "A", "acc": 0.80, "lat": 10.0}]
    front = compute_pareto_front(single, [("acc", "max"), ("lat", "min")])
    assert len(front) == 1
    assert front[0]["name"] == "A"


def test_pareto_dominance_2d():
    """Vérifie le filtrage de Pareto sur un cas d'école à 4 points.

    A : acc=80 %, lat=10 ms
    B : acc=75 %, lat=12 ms (dominé par A car moins précis et plus lent)
    C : acc=85 %, lat=15 ms (compromis : plus précis, mais plus lent)
    D : acc=70 %, lat=8 ms  (compromis : moins précis, mais plus rapide)

    Le front doit être {A, C, D}.
    """
    population = [
        {"name": "A", "acc": 0.80, "lat": 10.0},
        {"name": "B", "acc": 0.75, "lat": 12.0},
        {"name": "C", "acc": 0.85, "lat": 15.0},
        {"name": "D", "acc": 0.70, "lat": 8.0},
    ]
    objectives = [("acc", "max"), ("lat", "min")]

    assert is_dominated(population[1], population, objectives) is True
    assert is_dominated(population[0], population, objectives) is False

    front = compute_pareto_front(population, objectives)
    front_names = {p["name"] for p in front}
    assert front_names == {"A", "C", "D"}


def test_pareto_with_identical_points_keeps_both():
    """Deux points ayant exactement les mêmes métriques ne se dominent pas strictement."""
    population = [
        {"name": "A1", "acc": 0.80, "lat": 10.0},
        {"name": "A2", "acc": 0.80, "lat": 10.0},
    ]
    objectives = [("acc", "max"), ("lat", "min")]
    front = compute_pareto_front(population, objectives)
    assert len(front) == 2


def test_pareto_front_divergence_between_flops_and_latency():
    """Démontre qu'une architecture optimale en FLOPs peut être dominée en latence réelle.

    Scénario pédagogique :
    - Model_Shallow : acc=78 %, 50 M FLOPs, 5 ms de latence (convolutions larges, très vectorisables).
    - Model_Deep    : acc=78 %, 40 M FLOPs, 8 ms de latence (moins de FLOPs, mais 6 couches séquentielles lentes).

    Pour les FLOPs, Model_Deep domine Model_Shallow (même précision, moins de FLOPs).
    Pour la latence, Model_Shallow domine Model_Deep (même précision, plus rapide).
    """
    population = [
        {"name": "Shallow", "acc": 0.78, "flops": 50.0, "latency": 5.0},
        {"name": "Deep", "acc": 0.78, "flops": 40.0, "latency": 8.0},
    ]

    front_flops = compute_pareto_front(population, [("acc", "max"), ("flops", "min")])
    assert [p["name"] for p in front_flops] == ["Deep"]

    front_latency = compute_pareto_front(population, [("acc", "max"), ("latency", "min")])
    assert [p["name"] for p in front_latency] == ["Shallow"]


def test_is_dominated_rejects_unknown_direction():
    record = {"acc": 0.5}
    with pytest.raises(ValueError, match="direction inconnue"):
        is_dominated(record, [record], [("acc", "invalid")])
