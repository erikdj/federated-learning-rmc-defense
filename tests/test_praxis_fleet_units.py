import pytest

from praxis_exp.units import Unit, unit_id, expand_matrix, _token


def test_unit_id_is_deterministic_and_normalized():
    a = unit_id("Krum+TGE", "S4", "persistent_optimizer", 42)
    b = unit_id("Krum+TGE", "S4", "persistent_optimizer", 42)
    assert a == b
    assert a == "s4__krum_tge__persistent_optimizer__seed42"


def test_expand_matrix_is_complete_unique_and_ordered():
    units = expand_matrix(
        configs=["Krum", "TrustScore", "Krum+TGE", "FedAvg+TGE"],
        scenarios=["S0", "S1", "S2", "S3", "S4"],
        seeds=[42, 137, 256, 314, 500],
        mode="persistent_optimizer",
        max_per_client=2_000_000,
        rounds=50,
    )
    assert len(units) == 100
    assert len({u.unit_id for u in units}) == 100
    assert [u.array_index for u in units] == list(range(100))
    assert units[0].config == "Krum" and units[0].scenario == "S0" and units[0].seed == 42


def test_unit_carries_run_parameters():
    u = expand_matrix(["Krum"], ["S0"], [42], "persistent_optimizer", 2_000_000, 50)[0]
    assert u.max_per_client == 2_000_000 and u.rounds == 50 and u.mode == "persistent_optimizer"


def test_token_rejects_unexpected_characters():
    with pytest.raises(ValueError):
        _token("bad/name")          # slash is not [a-z0-9_]


def test_expand_matrix_rejects_empty_dimension():
    with pytest.raises(ValueError):
        expand_matrix([], ["S0"], [42], "persistent_optimizer", 2_000_000, 50)
    with pytest.raises(ValueError):
        expand_matrix(["Krum"], ["S0"], [], "persistent_optimizer", 2_000_000, 50)


# --- GWU-44 Lane B: optional repeats axis for same-seed replicates ----------
#
# The load-bearing invariant: a __rep{r} suffix appears IFF the repeat axis is
# active (repeats > 1). At repeats == 1 / absent, the emitted id string is
# byte-for-byte today's `{scenario}__{config}__{mode}__seed{seed}`. Any drift
# silently rewrites every existing matrix's unit identities (idempotent-skip
# markers, MLflow tags, S3 keys), so it is asserted here against literals.

# The exact pre-change expansion for this matrix, captured as literal strings
# so byte-identity is locked independently of the implementation.
_PRE_CHANGE_IDS = [
    "s0__krum__persistent_optimizer__seed42",
    "s0__krum__persistent_optimizer__seed137",
    "s4__krum__persistent_optimizer__seed42",
    "s4__krum__persistent_optimizer__seed137",
    "s0__trustscore__persistent_optimizer__seed42",
    "s0__trustscore__persistent_optimizer__seed137",
    "s4__trustscore__persistent_optimizer__seed42",
    "s4__trustscore__persistent_optimizer__seed137",
]


def test_unit_id_repeat_suffix():
    base = unit_id("Krum+TGE", "S4", "persistent_optimizer", 42)
    assert base == "s4__krum_tge__persistent_optimizer__seed42"
    # repeat >= 1 appends __rep{r}
    assert unit_id("Krum+TGE", "S4", "persistent_optimizer", 42, repeat=1) == base + "__rep1"
    assert unit_id("Krum+TGE", "S4", "persistent_optimizer", 42, repeat=3) == base + "__rep3"
    # the sentinel repeat=0 and an absent repeat are byte-for-byte today's id
    assert unit_id("Krum+TGE", "S4", "persistent_optimizer", 42, repeat=0) == base
    assert unit_id("Krum+TGE", "S4", "persistent_optimizer", 42) == base
    assert "__rep" not in unit_id("Krum+TGE", "S4", "persistent_optimizer", 42, repeat=0)


def test_expand_matrix_repeats_one_is_byte_identical():
    """repeats unset OR ==1 reproduces today's expansion exactly: same ids,
    same contiguous array_index, same count, and NO __rep token."""
    for kwargs in ({}, {"repeats": 1}):
        units = expand_matrix(
            ["Krum", "TrustScore"], ["S0", "S4"], [42, 137],
            "persistent_optimizer", 2_000_000, 50, **kwargs,
        )
        assert [u.unit_id for u in units] == _PRE_CHANGE_IDS
        assert [u.array_index for u in units] == list(range(8))
        assert all(u.repeat == 0 for u in units)
        assert all("__rep" not in u.unit_id for u in units)


def test_expand_matrix_repeats():
    """repeats=5 emits 5 distinct rep-suffixed units per (config,scenario,seed),
    innermost, with a contiguous array_index and repeat threaded onto the Unit."""
    units = expand_matrix(
        ["Krum"], ["S0"], [42], "persistent_optimizer", 2_000_000, 50, repeats=5,
    )
    assert [u.unit_id for u in units] == [
        "s0__krum__persistent_optimizer__seed42__rep1",
        "s0__krum__persistent_optimizer__seed42__rep2",
        "s0__krum__persistent_optimizer__seed42__rep3",
        "s0__krum__persistent_optimizer__seed42__rep4",
        "s0__krum__persistent_optimizer__seed42__rep5",
    ]
    assert [u.array_index for u in units] == [0, 1, 2, 3, 4]
    assert [u.repeat for u in units] == [1, 2, 3, 4, 5]
    assert len({u.unit_id for u in units}) == 5


def test_expand_matrix_repeats_multiplies_and_indexes_contiguously():
    """repeat is the INNERMOST axis; array_index stays a single 0..N-1 counter
    across the full (config x scenario x seed x repeat) product."""
    units = expand_matrix(
        ["Krum", "TrustScore"], ["S0", "S4"], [42, 137],
        "persistent_optimizer", 2_000_000, 50, repeats=3,
    )
    assert len(units) == 2 * 2 * 2 * 3  # 24
    assert len({u.unit_id for u in units}) == 24
    assert [u.array_index for u in units] == list(range(24))
    # first three units share (config,scenario,seed), differing only by rep
    assert units[0].unit_id == "s0__krum__persistent_optimizer__seed42__rep1"
    assert units[1].unit_id == "s0__krum__persistent_optimizer__seed42__rep2"
    assert units[2].unit_id == "s0__krum__persistent_optimizer__seed42__rep3"
    # the fourth advances the seed axis, rep restarts at 1
    assert units[3].unit_id == "s0__krum__persistent_optimizer__seed137__rep1"


def test_expand_matrix_rejects_non_positive_repeats():
    with pytest.raises(ValueError):
        expand_matrix(["Krum"], ["S0"], [42], "persistent_optimizer", 2_000_000, 50, repeats=0)
    with pytest.raises(ValueError):
        expand_matrix(["Krum"], ["S0"], [42], "persistent_optimizer", 2_000_000, 50, repeats=-1)


def test_unit_id_rejects_negative_repeat():
    """A NEGATIVE repeat must fail fast, not fall through to the base id — a
    corrupt/tampered manifest carrying repeat=-1 would otherwise silently ALIAS
    onto the un-suffixed unit's identity (marker / MLflow tag / S3 key collision
    with a real non-replicate unit)."""
    with pytest.raises(ValueError):
        unit_id("Krum", "S0", "persistent_optimizer", 42, repeat=-1)


def test_unit_property_rejects_negative_repeat():
    """Same guard on the property path (Unit(**u).unit_id), which is how a
    manifest-loaded unit computes its id."""
    u = Unit("Krum", "S0", "persistent_optimizer", 42, 2_000_000, 50, 0, repeat=-1)
    with pytest.raises(ValueError):
        _ = u.unit_id
