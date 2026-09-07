"""Durable per-client resampling manifest + arm-compliance assertions (§5).

Verifies the manifest row schema and the LOUD compliance checks that FAIL a unit
whose recorded arm config disagrees with what actually executed: step-cap
equality, weight-mode reported mass, declared-arm consistency, and semantic
after-class-count contracts.
"""
import pytest

from flowerfl.resampling_manifest import (
    build_manifest_row,
    assert_arm_compliance,
    assert_manifest_complete,
    MANIFEST_FIELDS,
)
from flowerfl.smote_resampler import semantic_attack_target_count


def _prep_info(**over):
    base = {
        "n_orig": 100, "n_resampled": 100,
        "n_benign_before": 90, "n_attack_before": 10,
        "n_benign_after": 90, "n_attack_after": 10,
        "sampler_status": "off", "skip_reason": None, "k_eff": 0,
        "variant": None, "target_fraction": None,
    }
    base.update(over)
    return base


def _row(prep_info=None, **over):
    kwargs = dict(
        partition_id=3, arm="off", weight_mode="resampled", update_match=False,
        num_examples=100, max_steps=None, actual_steps=None, semantic_policy=False,
        prep_info=prep_info or _prep_info(),
    )
    kwargs.update(over)
    return build_manifest_row(**kwargs)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_manifest_row_has_all_fields():
    row = _row()
    for field in MANIFEST_FIELDS:
        assert field in row, f"missing manifest field {field}"


# ---------------------------------------------------------------------------
# Compliant rows pass
# ---------------------------------------------------------------------------

def test_off_arm_row_is_compliant():
    assert_arm_compliance(_row())  # no raise


def test_matched_original_smote_row_is_compliant():
    # smote@0.5 applied, semantic: 90 benign / 10 attack -> attack grown to 90.
    prep = _prep_info(
        n_resampled=180, n_benign_after=90, n_attack_after=90,
        sampler_status="applied", k_eff=5, variant="smote", target_fraction=0.5,
    )
    row = _row(
        prep_info=prep, arm="smote@0.5", weight_mode="original", update_match=True,
        num_examples=100, max_steps=160, actual_steps=160, semantic_policy=True,
    )
    assert_arm_compliance(row, expected={
        "variant": "smote", "target_fraction": 0.5,
        "weight_mode": "original", "update_match": True,
    })


# ---------------------------------------------------------------------------
# (1) update-matching cap equality
# ---------------------------------------------------------------------------

def test_actual_steps_below_cap_fails():
    row = _row(update_match=True, max_steps=160, actual_steps=153)
    with pytest.raises(ValueError, match="actual_steps=153 != max_steps=160"):
        assert_arm_compliance(row)


# ---------------------------------------------------------------------------
# (2) weight-mode reported mass
# ---------------------------------------------------------------------------

def test_original_weight_mode_must_report_n_orig():
    prep = _prep_info(n_resampled=180, n_benign_after=90, n_attack_after=90,
                      sampler_status="applied", variant="random_over", target_fraction=0.5)
    row = _row(prep_info=prep, weight_mode="original", num_examples=180,  # wrong: should be n_orig=100
               semantic_policy=True, arm="random_over@0.5")
    with pytest.raises(ValueError, match="weight-mode=original but num_examples=180 != n_orig=100"):
        assert_arm_compliance(row)


def test_resampled_weight_mode_must_report_n_resampled():
    prep = _prep_info(n_resampled=180, n_benign_after=90, n_attack_after=90,
                      sampler_status="applied", variant="random_over", target_fraction=0.5)
    row = _row(prep_info=prep, weight_mode="resampled", num_examples=100,  # wrong: should be 180
               semantic_policy=True, arm="random_over@0.5")
    with pytest.raises(ValueError, match="weight-mode=resampled but num_examples=100 != n_resampled=180"):
        assert_arm_compliance(row)


# ---------------------------------------------------------------------------
# (3) declared-arm consistency
# ---------------------------------------------------------------------------

def test_declared_arm_mismatch_fails():
    row = _row()
    with pytest.raises(ValueError, match="declared variant='random_under' disagrees"):
        assert_arm_compliance(row, expected={
            "variant": "random_under", "target_fraction": None,
            "weight_mode": "resampled", "update_match": False,
        })


# ---------------------------------------------------------------------------
# (4) semantic after-class-count contracts
# ---------------------------------------------------------------------------

def test_semantic_oversampler_wrong_after_count_fails():
    # attack must be grown to ceil(0.47/0.53 * 90) = 80; a row claiming 90 fails.
    assert semantic_attack_target_count(0.47, 90) == 80
    prep = _prep_info(n_resampled=180, n_benign_after=90, n_attack_after=90,
                      sampler_status="applied", variant="smote", target_fraction=0.47)
    row = _row(prep_info=prep, weight_mode="resampled", num_examples=180,
               semantic_policy=True, arm="smote@0.47")
    with pytest.raises(ValueError, match="expected benign=90, attack=80; got benign=90, attack=90"):
        assert_arm_compliance(row)


def test_semantic_oversampler_growing_benign_fails():
    # Benign must be held FIXED; a row where benign changed fails loudly.
    prep = _prep_info(n_resampled=180, n_benign_after=100, n_attack_after=90,
                      sampler_status="applied", variant="random_over", target_fraction=0.5)
    row = _row(prep_info=prep, weight_mode="resampled", num_examples=180,
               semantic_policy=True, arm="random_over@0.5")
    with pytest.raises(ValueError, match="expected benign=90, attack=90; got benign=100"):
        assert_arm_compliance(row)


def test_semantic_undersampler_shrinks_benign_ok():
    # 90 benign / 10 attack -> benign shrunk to 10 (attack fraction 0.5).
    prep = _prep_info(n_resampled=20, n_benign_after=10, n_attack_after=10,
                      sampler_status="applied", variant="random_under", target_fraction=0.5)
    row = _row(prep_info=prep, weight_mode="resampled", num_examples=20,
               semantic_policy=True, arm="random_under@0.5")
    assert_arm_compliance(row)  # no raise


def test_semantic_undersampler_touching_attack_fails():
    prep = _prep_info(n_resampled=20, n_benign_after=10, n_attack_after=9,  # attack changed
                      sampler_status="applied", variant="random_under", target_fraction=0.5)
    row = _row(prep_info=prep, weight_mode="resampled", num_examples=20,
               semantic_policy=True, arm="random_under@0.5")
    with pytest.raises(ValueError, match="attack must be untouched"):
        assert_arm_compliance(row)


def test_skip_must_leave_counts_untouched():
    prep = _prep_info(sampler_status="skipped", skip_reason="attack_at_or_above_target",
                      variant="smote", target_fraction=0.5,
                      n_benign_after=80, n_attack_after=10)  # benign changed on a skip
    row = _row(prep_info=prep, arm="smote@0.5", semantic_policy=True)
    with pytest.raises(ValueError, match="sampler_status=skipped but after-counts"):
        assert_arm_compliance(row)


def test_legacy_min_max_applied_skips_semantic_contract():
    # semantic_policy=False -> after-counts are NOT held to the §6 contract.
    prep = _prep_info(n_resampled=180, n_benign_after=90, n_attack_after=90,
                      sampler_status="applied", variant="smote", target_fraction=0.5)
    row = _row(prep_info=prep, weight_mode="resampled", num_examples=180,
               semantic_policy=False, arm="smote@balanced")
    assert_arm_compliance(row)  # no raise despite not matching semantic ceil


# ---------------------------------------------------------------------------
# Unit-level completeness gate (assert_manifest_complete)
# ---------------------------------------------------------------------------

def _complete_row(pid):
    """A manifest row carrying every MANIFEST_FIELDS key (schema-complete)."""
    row = {k: 0 for k in MANIFEST_FIELDS}
    row["partition_id"] = pid
    return row


def test_manifest_complete_passes_on_exact_coverage():
    rows = {0: _complete_row(0), 1: _complete_row(1)}
    assert_manifest_complete(rows, {0, 1})  # no raise


def test_manifest_complete_fails_on_missing_partition():
    rows = {0: _complete_row(0)}
    with pytest.raises(ValueError, match="missing"):
        assert_manifest_complete(rows, {0, 1})


def test_manifest_complete_fails_on_unexpected_partition():
    rows = {0: _complete_row(0), 5: _complete_row(5)}
    with pytest.raises(ValueError, match="never dispatched"):
        assert_manifest_complete(rows, {0})


def test_manifest_complete_fails_on_missing_field():
    bad = _complete_row(0)
    del bad["k_eff"]
    with pytest.raises(ValueError, match="k_eff"):
        assert_manifest_complete({0: bad}, {0})


def test_manifest_complete_message_lists_all_violation_classes():
    rows = {0: _complete_row(0), 9: _complete_row(9)}  # 9 unexpected
    del rows[0]["k_eff"]                                # 0 schema-incomplete
    with pytest.raises(ValueError) as ei:
        assert_manifest_complete(rows, {0, 1})         # 1 missing
    msg = str(ei.value)
    assert "1" in msg and "9" in msg and "k_eff" in msg


def test_manifest_complete_unresolved_cids_none_or_empty_ok():
    rows = {0: _complete_row(0)}
    assert_manifest_complete(rows, {0}, unresolved_cids=None)   # no raise
    assert_manifest_complete(rows, {0}, unresolved_cids=set())  # no raise


def test_manifest_complete_fails_on_unresolved_cids_alone():
    # Coverage is exact and schema-complete, but a discovery-round client never
    # resolved to a partition -> FOURTH violation class fires on its own.
    rows = {0: _complete_row(0)}
    with pytest.raises(ValueError, match="never resolved"):
        assert_manifest_complete(rows, {0}, unresolved_cids={"rawX"})


def test_manifest_complete_unresolved_cid_is_named():
    rows = {0: _complete_row(0)}
    with pytest.raises(ValueError) as ei:
        assert_manifest_complete(rows, {0}, unresolved_cids={"rawZ"})
    assert "rawZ" in str(ei.value)


def test_manifest_complete_message_combines_unresolved_with_others():
    rows = {0: _complete_row(0), 9: _complete_row(9)}  # 9 unexpected
    with pytest.raises(ValueError) as ei:
        assert_manifest_complete(rows, {0, 1}, unresolved_cids={"rawZ"})  # 1 missing
    msg = str(ei.value)
    assert "1" in msg and "9" in msg and "rawZ" in msg
