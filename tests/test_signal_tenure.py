import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_tenure_counts_rounds_since_identity_first_seen():
    from flowerfl.scenario_strategy import compute_tenure
    seen = {}
    assert compute_tenure("client_0", 1, seen) == 1
    assert compute_tenure("client_0", 2, seen) == 2
    assert compute_tenure("client_5_new1", 3, seen) == 1
    assert compute_tenure("client_0", 3, seen) == 3
