"""parse_eval_trajectory: F1/Acc/Loss are always parsed; Prec/Rec are appended
by the new-image ScenarioStrategy eval line and captured when present, staying
backward-compatible with older logs that omit them."""
from run_phase4_flower import parse_eval_trajectory


def test_parses_legacy_line_without_prec_rec():
    out = "[ScenarioStrategy] Round 3 eval: F1=0.9698 Acc=0.9701 Loss=0.1229\n"
    assert parse_eval_trajectory(out) == [
        {"round": 3, "f1": 0.9698, "accuracy": 0.9701, "loss": 0.1229}
    ]


def test_parses_new_line_with_prec_rec():
    out = ("[ScenarioStrategy] Round 5 eval: F1=0.9636 Acc=0.9638 Loss=0.1830 "
           "Prec=0.9642 Rec=0.9632\n")
    traj = parse_eval_trajectory(out)
    assert traj[0]["round"] == 5 and traj[0]["f1"] == 0.9636
    assert traj[0]["precision"] == 0.9642
    assert traj[0]["recall"] == 0.9632


def test_mixed_lines_parse_each_correctly():
    out = ("[ScenarioStrategy] Round 1 eval: F1=0.1 Acc=0.2 Loss=0.3\n"
           "[ScenarioStrategy] Round 2 eval: F1=0.4 Acc=0.5 Loss=0.6 Prec=0.7 Rec=0.8\n")
    traj = parse_eval_trajectory(out)
    assert len(traj) == 2
    assert "precision" not in traj[0]  # legacy line: no prec/rec fabricated
    assert traj[1]["precision"] == 0.7 and traj[1]["recall"] == 0.8


def test_parses_per_class_fields_when_present():
    """Stage-F: the eval line appends per-class benign/attack metrics; the
    parser captures all six into append-only trajectory keys."""
    out = ("[ScenarioStrategy] Round 7 eval: F1=0.90 Acc=0.91 Loss=0.20 "
           "Prec=0.92 Rec=0.93 AttP=0.60 AttR=0.75 AttF1=0.67 "
           "BenP=0.80 BenR=0.67 BenF1=0.73\n")
    traj = parse_eval_trajectory(out)
    row = traj[0]
    assert row["round"] == 7
    assert row["precision"] == 0.92 and row["recall"] == 0.93
    assert row["attack_precision"] == 0.60
    assert row["attack_recall"] == 0.75
    assert row["attack_f1"] == 0.67
    assert row["benign_precision"] == 0.80
    assert row["benign_recall"] == 0.67
    assert row["benign_f1"] == 0.73


def test_legacy_and_prec_rec_lines_omit_per_class_keys():
    """Append-only: lines without the per-class suffix must not fabricate the
    new keys, so old logs replay identically."""
    out = ("[ScenarioStrategy] Round 1 eval: F1=0.1 Acc=0.2 Loss=0.3\n"
           "[ScenarioStrategy] Round 2 eval: F1=0.4 Acc=0.5 Loss=0.6 Prec=0.7 Rec=0.8\n")
    traj = parse_eval_trajectory(out)
    for row in traj:
        assert "attack_recall" not in row
        assert "benign_precision" not in row
