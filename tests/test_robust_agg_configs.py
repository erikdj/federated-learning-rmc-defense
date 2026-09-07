import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

@pytest.mark.parametrize("name,frag", [("FedMedian","FedMedian"), ("FedTrimmedAvg","FedTrimmedAvg")])
def test_robust_agg_supported_and_routes(name, frag):
    from run_phase4_flower import SUPPORTED_CONFIGS, build_strategy_for_config
    assert name in SUPPORTED_CONFIGS
    token, cfg = build_strategy_for_config(name)
    assert frag in token.__name__
