"""Normalization-leak (m1) opt-in fix — fleet-reachable, default-OFF.

A normalization audit confirmed a PRE-EXISTING, all-arms leak: load_data fits the
per-client Z-score on the FULL capped dataframe (train+val+test) BEFORE the
80/10/10 split, so local val/test feature distributions influence the training
transform. This module locks the fix behind ``--normalize-train-only`` /
run_extras ``normalize_train_only`` (default OFF) and asserts:

  1. flag OFF -> the normalization statistics are BYTE-IDENTICAL to the sealed
     incumbent (all-rows fit), so the canonical pre-registered pipeline is
     untouched;
  2. flag ON  -> statistics are fitted EXCLUSIVELY on the training rows of the
     fixed seed-42 split (the SAME indices random_split assigns downstream);
  3. the reachability chain (matrix_doc -> manifest run_extras -> entrypoint
     argv -> runner CLI -> run-config) carries the knob end-to-end and stays
     byte-identical when the knob is absent or at its incumbent value.

Mirrors tests/test_smote_fleet.py sections 9-10 for the fleet-path assertions.
"""
import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ===========================================================================
# 1. Core statistic: flag-OFF byte-identity + flag-ON train-only correctness
# ===========================================================================

def _rand_X(rows=200, cols=7, seed=0):
    rng = np.random.default_rng(seed)
    # Deliberately non-uniform across the row axis so an all-rows fit and a
    # train-only fit MUST differ (guards against a false-pass where the two
    # happen to coincide). One constant (zero-variance) column exercises the
    # std==0 -> 1.0 guard.
    X = rng.normal(loc=3.0, scale=5.0, size=(rows, cols)).astype(np.float32)
    X[:, 0] = 7.0  # zero-variance column
    return X


def _incumbent_stats(X):
    """The exact sealed computation (task.py pre-fix): fit on ALL rows."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def test_zscore_stats_flag_off_is_byte_identical_to_incumbent():
    from flowerfl.task import compute_zscore_stats
    X = _rand_X()
    exp_mean, exp_std = _incumbent_stats(X.copy())
    mean, std = compute_zscore_stats(X, train_split=0.8, normalize_train_only=False)
    # Bit-exact: the default path must reproduce every sealed run.
    assert np.array_equal(mean, exp_mean)
    assert np.array_equal(std, exp_std)


def test_zscore_stats_flag_off_does_not_mutate_input():
    from flowerfl.task import compute_zscore_stats
    X = _rand_X()
    before = X.copy()
    compute_zscore_stats(X, train_split=0.8, normalize_train_only=False)
    assert np.array_equal(X, before)  # immutability: no in-place edit of X


def test_zscore_stats_flag_on_fits_train_rows_only():
    from flowerfl.task import compute_zscore_stats
    X = _rand_X()
    n_total = X.shape[0]
    n_train = int(0.8 * n_total)
    perm = torch.randperm(
        n_total, generator=torch.Generator().manual_seed(42)
    ).numpy()
    train_rows = X[perm[:n_train]]
    exp_mean = train_rows.mean(axis=0)
    exp_std = train_rows.std(axis=0)
    exp_std[exp_std == 0] = 1.0

    mean, std = compute_zscore_stats(X, train_split=0.8, normalize_train_only=True)
    assert np.array_equal(mean, exp_mean)
    assert np.array_equal(std, exp_std)


def test_zscore_stats_on_and_off_differ():
    """The two modes must produce DIFFERENT stats on realistic data — otherwise
    the fix is a no-op and the byte-identity test above is meaningless."""
    from flowerfl.task import compute_zscore_stats
    X = _rand_X(seed=7)
    off_mean, _ = compute_zscore_stats(X, 0.8, normalize_train_only=False)
    on_mean, _ = compute_zscore_stats(X, 0.8, normalize_train_only=True)
    assert not np.array_equal(off_mean, on_mean)


def test_zscore_stats_train_indices_match_random_split():
    """The train-only fit MUST use exactly the rows random_split (seed 42)
    assigns to the training subset downstream in load_data — no off-by-one, no
    divergent permutation. We reconstruct random_split's assignment and confirm
    the stats equal a fit over those precise rows."""
    from flowerfl.task import compute_zscore_stats
    X = _rand_X(seed=11)
    n_total = X.shape[0]
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X), torch.zeros(n_total, dtype=torch.long)
    )
    train_set, _, _ = torch.utils.data.random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    split_train_idx = np.array(train_set.indices)
    exp_mean = X[split_train_idx].mean(axis=0)

    mean, _ = compute_zscore_stats(X, 0.8, normalize_train_only=True)
    assert np.allclose(mean, exp_mean)


# ===========================================================================
# 2. runner CLI: flag -> run-config injection (default OFF preserved)
# ===========================================================================

def test_leakage_run_config_from_cli_off_is_empty():
    from run_phase4_flower import leakage_run_config_from_cli
    assert leakage_run_config_from_cli(False) == {}


def test_leakage_run_config_from_cli_on_injects_hyphenated_key():
    from run_phase4_flower import leakage_run_config_from_cli
    assert leakage_run_config_from_cli(True) == {"normalize-train-only": True}


def test_runner_parser_defaults_normalize_train_only_off():
    from run_phase4_flower import add_leakage_cli_args
    p = argparse.ArgumentParser()
    add_leakage_cli_args(p)
    args = p.parse_args([])
    assert args.normalize_train_only is False


def test_runner_parser_accepts_normalize_train_only_flag():
    from run_phase4_flower import add_leakage_cli_args
    p = argparse.ArgumentParser()
    add_leakage_cli_args(p)
    args = p.parse_args(["--normalize-train-only"])
    assert args.normalize_train_only is True


def test_leakage_provenance_absent_is_incumbent():
    from run_phase4_flower import normalize_train_only_provenance
    assert normalize_train_only_provenance({}) == {"normalize_train_only": False}


def test_leakage_provenance_present_true():
    from run_phase4_flower import normalize_train_only_provenance
    assert normalize_train_only_provenance(
        {"normalize-train-only": True}) == {"normalize_train_only": True}


# ===========================================================================
# 3. entrypoint: argv byte-identity without run_extras; flag with it
#    (mirrors test_smote_fleet.py sections 9-10)
# ===========================================================================

def _unit():
    from praxis_exp.units import Unit
    return Unit("Krum+TGE", "S4", "persistent_optimizer", 42, 2_000_000, 50, 0)


def test_leakage_argv_absent_or_incumbent_is_empty():
    from docker.entrypoint import leakage_argv_from_run_extras
    assert leakage_argv_from_run_extras(None) == []
    assert leakage_argv_from_run_extras({}) == []
    assert leakage_argv_from_run_extras({"normalize_train_only": False}) == []


def test_leakage_argv_maps_flag():
    from docker.entrypoint import leakage_argv_from_run_extras
    assert leakage_argv_from_run_extras(
        {"normalize_train_only": True}) == ["--normalize-train-only"]


@pytest.mark.parametrize("bad", ["ture", "1", "yes", 1])
def test_leakage_argv_raises_on_garbage_bool(bad):
    from docker.entrypoint import leakage_argv_from_run_extras
    with pytest.raises(ValueError, match="normalize_train_only"):
        leakage_argv_from_run_extras({"normalize_train_only": bad})


def test_runner_argv_byte_identical_without_leakage_extra():
    from docker.entrypoint import runner_argv
    u = _unit()
    base = runner_argv(u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-060/x"))
    for extras in (None, {}, {"normalize_train_only": False}):
        assert runner_argv(u, scenario_dir="rmc/scenarios",
                           out_dir=Path("results/EXP-060/x"),
                           run_extras=extras) == base


def test_runner_argv_appends_leakage_flag_with_run_extras():
    from docker.entrypoint import runner_argv
    u = _unit()
    base = runner_argv(u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-060/x"))
    argv = runner_argv(
        u, scenario_dir="rmc/scenarios", out_dir=Path("results/EXP-060/x"),
        run_extras={"normalize_train_only": True},
    )
    assert argv[:len(base)] == base
    assert "--normalize-train-only" in argv[len(base):]


# ===========================================================================
# 4. matrix_doc: run_extras allowlist + strict bool coercion at pre-reg parse
# ===========================================================================

_DOC = textwrap.dedent('''\
    ---
    exp_id: EXP-060
    slug: leak-probe
    hypothesis: H2
    methodology_version: v1.30
    matrix:
      defenses: [Krum+TGE]
      scenarios: [control_honest, S4]
      seeds: [42]
      mode: persistent_optimizer
      max_per_client: 2000000
      rounds: 50
    batch:
      job_queue: praxis-spot-queue
      job_definition: praxis-flowerfl-unit
    {run_extras}---
    Normalization-leak probe.
    ''')


def _doc_text(block=""):
    return _DOC.format(run_extras=block)


def test_matrix_doc_normalize_train_only_parsed_and_normalized(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix
    p = tmp_path / "d.md"
    p.write_text(_doc_text("run_extras:\n  normalize_train_only: true\n"))
    assert parse_matrix(p).run_extras["normalize_train_only"] is True


def test_matrix_doc_normalize_train_only_bad_bool_rejected(tmp_path):
    from praxis_exp.matrix_doc import parse_matrix, MatrixDocError
    p = tmp_path / "d.md"
    p.write_text(_doc_text('run_extras:\n  normalize_train_only: "ture"\n'))
    with pytest.raises(MatrixDocError, match="normalize_train_only"):
        parse_matrix(p)


def test_matrix_doc_normalize_train_only_full_chain_through_real_parser(tmp_path):
    """End-to-end: run_extras -> entrypoint argv -> REAL runner parser ->
    run-config override, with nothing dropped along the way."""
    from praxis_exp.matrix_doc import parse_matrix
    from docker.entrypoint import leakage_argv_from_run_extras
    from run_phase4_flower import add_leakage_cli_args, leakage_run_config_from_cli
    p = tmp_path / "d.md"
    p.write_text(_doc_text("run_extras:\n  normalize_train_only: true\n"))
    extras = parse_matrix(p).run_extras
    argv = leakage_argv_from_run_extras(extras)
    parser = argparse.ArgumentParser()
    add_leakage_cli_args(parser)
    args = parser.parse_args(argv)
    assert leakage_run_config_from_cli(args.normalize_train_only) == {
        "normalize-train-only": True
    }


# ===========================================================================
# 5. Prewarm key-parity: the driver primes / validates the
#    SAME resample cache key every Ray worker looks up. Omitting the mode
#    would prime the LEAK-ON key while leak-free workers all miss and
#    concurrently recompute the big train-only resample (EXP-020-class memory
# failure; same family as the semantic-target prewarm bug). Mirrors
#    tests/test_pr33_review_fixes.py P1-2.
# ===========================================================================

import flowerfl.task as task_module


def test_resample_cache_key_partitions_on_normalize_train_only():
    k_leak_on = task_module._resample_cache_key(
        "cic", 0, 32, 0.8, 0.1, "smote", "balanced", 42, normalize_train_only=False)
    k_leak_free = task_module._resample_cache_key(
        "cic", 0, 32, 0.8, 0.1, "smote", "balanced", 42, normalize_train_only=True)
    assert k_leak_on != k_leak_free
    # append-only-when-True: the leak-on key carries NO token (byte-identical to
    # the pre-fix key), the leak-free key ends with the partition token.
    assert "normalize_train_only" not in k_leak_on
    assert k_leak_free[-1] == "normalize_train_only"


def test_resample_cache_path_flag_off_byte_identical_to_incumbent():
    """Flag-OFF durability-pass key MUST equal the incumbent (pre-fix) key: the
    default arg appends no token, so the driver validates exactly the leak-on
    path today's sweep already primes."""
    default_path = task_module.resample_cache_path(
        "cic", 0, smote_variant="smote", smote_target="balanced", smote_seed=42)
    explicit_off = task_module.resample_cache_path(
        "cic", 0, smote_variant="smote", smote_target="balanced", smote_seed=42,
        normalize_train_only=False)
    incumbent_key = task_module._resample_cache_key(
        "cic", 0, 32, 0.8, 0.1, "smote", "balanced", 42)  # no normalize arg
    assert default_path == explicit_off
    assert default_path == task_module._resample_disk_path(incumbent_key)


def test_resample_cache_path_threads_normalize_train_only_to_key():
    """Driver-vs-worker key parity: the driver durability pass (resample_cache_path)
    resolves the SAME key the worker (_resample_cache_key, called inside
    load_data) looks up for a leak-free resample."""
    driver_path = task_module.resample_cache_path(
        "cic", 0, smote_variant="smote", smote_target="balanced", smote_seed=42,
        normalize_train_only=True)
    worker_key = task_module._resample_cache_key(
        "cic", 0, 32, 0.8, 0.1, "smote", "balanced", 42, normalize_train_only=True)
    assert driver_path == task_module._resample_disk_path(worker_key)
    # leak-free and leak-on resolve to DIFFERENT durable paths (or both None only
    # when the disk layer is unavailable — then the key-level test is the proof).
    leak_on_path = task_module.resample_cache_path(
        "cic", 0, smote_variant="smote", smote_target="balanced", smote_seed=42,
        normalize_train_only=False)
    if driver_path is not None or leak_on_path is not None:
        assert driver_path != leak_on_path


def test_prewarm_threads_normalize_train_only_to_both_calls():
    """Guard against the exact regression: the prewarm must read the
    normalize-train-only run-config key AND thread it into BOTH the load_data
    priming call and the resample_cache_path durability pass."""
    import inspect
    from run_phase4_flower import _prewarm_resample_cache
    src = inspect.getsource(_prewarm_resample_cache)
    assert "normalize-train-only" in src  # reads the run-config key
    # threaded into both the load_data prime and the resample_cache_path
    # durability pass (two occurrences of the keyword-arg pass-through).
    assert src.count("normalize_train_only=normalize_train_only") == 2
