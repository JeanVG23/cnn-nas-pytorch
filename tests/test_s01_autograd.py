import pytest
import torch

from steps.s01_autograd import (
    W1_INIT,
    W2_INIT,
    accumulated_w1_grad,
    autograd_grads,
    descend,
    forward,
    graph_nodes,
    manual_grads,
)


def test_manual_gradient_matches_autograd():
    manual = manual_grads(W1_INIT, W2_INIT)
    auto = autograd_grads(W1_INIT, W2_INIT)
    assert auto == pytest.approx(manual, rel=1e-9)


def test_two_backwards_without_zeroing_double_the_gradient():
    once = autograd_grads(W1_INIT, W2_INIT)[0]
    twice = accumulated_w1_grad(W1_INIT, W2_INIT, backwards=2)
    assert twice == pytest.approx(2 * once, rel=1e-9)


def test_graph_contains_tanh_and_pow_nodes():
    w1 = torch.tensor(W1_INIT, dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(W2_INIT, dtype=torch.float64, requires_grad=True)
    _, _, loss = forward(w1, w2)
    nodes = graph_nodes(loss)
    assert "TanhBackward0" in nodes
    assert "PowBackward0" in nodes


def test_descent_reduces_loss():
    losses = descend(W1_INIT, W2_INIT, lr=0.1, steps=50)
    assert len(losses) == 50
    assert losses[-1] < losses[0]
