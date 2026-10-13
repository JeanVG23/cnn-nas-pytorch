from common.data import split_indices


def test_split_indices_are_disjoint_and_cover_everything():
    train_idx, val_idx = split_indices(100, 20)
    assert len(train_idx) == 80
    assert len(val_idx) == 20
    assert set(train_idx).isdisjoint(val_idx)
    assert sorted(train_idx + val_idx) == list(range(100))


def test_split_indices_is_deterministic_and_depends_on_split_seed():
    assert split_indices(100, 20) == split_indices(100, 20)
    assert split_indices(100, 20, split_seed=1) != split_indices(100, 20, split_seed=2)
