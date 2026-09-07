from pathlib import Path
from unittest.mock import MagicMock


def test_list_writes_index_md(tmp_path):
    from praxis_exp.listing import list_experiments
    exp_dir = tmp_path / "docs" / "experiments"
    exp_dir.mkdir(parents=True)
    fake_client = MagicMock()
    fake_client.search_runs.return_value = [
        {"exp_id": "EXP-001", "slug": "test", "status": "FINISHED",
         "methodology_version": "v1.2", "criteria_ok": "true"},
    ]
    list_experiments(tmp_path, _client=fake_client)
    idx = exp_dir / "INDEX.md"
    assert idx.exists()
    txt = idx.read_text()
    assert "EXP-001" in txt
