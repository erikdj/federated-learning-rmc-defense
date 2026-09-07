import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_select_threshold_holds_fpr_on_honest():
    from h2_threshold_pipeline import select_threshold
    honest = [0.9, 0.92, 0.88, 0.95, 0.91, 0.93, 0.89, 0.9, 0.94, 0.9]
    th = select_threshold(honest_scores=honest, target_fpr=0.10)
    assert sum(1 for s in honest if s < th) <= 1


def test_freeze_and_apply_roundtrip(tmp_path):
    from h2_threshold_pipeline import freeze_thresholds, load_thresholds
    p = tmp_path / "th.json"
    freeze_thresholds({"Krum": 0.5, "TGE": 0.7}, str(p))
    assert load_thresholds(str(p)) == {"Krum": 0.5, "TGE": 0.7}
