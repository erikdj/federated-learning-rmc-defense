import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_adversarial_set_from_schedule_includes_dormant_rounds():
    from flowerfl.scenario_strategy import adversarial_identities
    schedule = [
        {"rounds": [1, 2], "participants": ["c0","c1"], "attacks": {}},
        {"rounds": [3, 4], "participants": ["c0","c1"], "attacks": {"c0": {"type": "alie"}}},
    ]
    adv = adversarial_identities(schedule)
    assert "c0" in adv and "c1" not in adv
