import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_coldstart_filter_keeps_only_first_k_tenure_rows():
    from compute_recall_fpr import filter_scope
    rows = [{"tenure": t, "score": 0.5, "malicious_gt": True} for t in [1, 2, 3, 4, 5]]
    cs = filter_scope(rows, scope="coldstart", k=3)
    assert [r["tenure"] for r in cs] == [1, 2, 3]
    allr = filter_scope(rows, scope="all", k=3)
    assert len(allr) == 5
