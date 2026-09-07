import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_train_eval_uses_disjoint_seed_sets():
    from h1_signal_family_eval import split_by_seed
    rows = [{"seed": s, "x": 1} for s in [42,137,256] for _ in range(3)] + \
           [{"seed": s, "x": 1} for s in [1009,1733] for _ in range(3)]
    train, ev = split_by_seed(rows, train_seeds=[42,137,256], eval_seeds=[1009,1733])
    assert {r["seed"] for r in train} == {42,137,256}
    assert {r["seed"] for r in ev} == {1009,1733}
    assert not ({r["seed"] for r in train} & {r["seed"] for r in ev})
