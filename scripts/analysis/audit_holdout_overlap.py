"""Audit row-overlap between the server-side evaluation holdout and per-client
TRAIN splits — the evaluation-hygiene question the advisor raised (test leakage).

Canonical dev config audited:
    train dataset      = edge_full_20_rmc  (21 partitions, data/edge_full_20)
    eval  dataset      = edge_full_20      (holdout; server_app maps _rmc -> base)
    seed               = 42
    max_per_client     = 2_000_000  (MAX_SAMPLES_PER_CLIENT at the honest-control cap)
    holdout spc        = 2000, seed 42     (server_app.py:155 defaults)

Method (index-only; row SELECTION depends solely on labels + row counts, so we
read ONLY the label column — faithful and memory-frugal, one client at a time):

  * Holdout indices are reconstructed exactly as rmc/fixed_eval.py::_build_holdout
    draws them: a SINGLE np.random.RandomState(42) shared across the 20 eval files
    in file order, per-class stratified rng.choice. We do NOT copy the feature/
    normalization work (it never touches the RNG). Fidelity is checked by
    constructing the REAL FixedEvalManager and asserting our reconstructed holdout
    matches its total sample count and positive count exactly.

  * Train indices are reconstructed exactly as flowerfl/task.py::load_data produces
    them at the 2M cap: stratified cap sampling via groupby(label).sample(
    random_state=42+client_idx) + concat + reset_index (only for the 3 clients whose
    row count exceeds 2M), then random_split into train/val/test with
    torch.Generator().manual_seed(42). torch.random_split takes the first
    ``train_size`` entries of randperm(total, generator=g); we reproduce that and
    map the capped-frame positions back to original parquet row positions.

We cannot import load_data directly (it returns DataLoaders, not indices), so the
train path is a line-faithful re-implementation, not a copy — flagged as the one
residual fidelity risk in the memo.

Outputs (all under results/20260727/holdout_overlap_audit/):
    overlap.json           machine-readable per-client + aggregate numbers
    OVERLAP_AUDIT_MEMO.md   method + config + numbers + implication

Run:
    conda run -n flowerfl python scripts/analysis/audit_holdout_overlap.py \
        > results/20260727/holdout_overlap_audit/run.log 2>&1
"""
import gc
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flowerfl.task import get_dataset_config, is_full_dataset  # noqa: E402

# ---- audited configuration (canonical dev H2 honest-control cell) ----------
TRAIN_DATASET = "edge_full_20_rmc"
EVAL_DATASET = "edge_full_20"          # server_app.py:153 -> dataset.replace("_rmc","")
SEED = 42
MAX_SAMPLES_PER_CLIENT = 2_000_000     # honest-control cap (EXP-010 matrix: 2000000)
HOLDOUT_SPC = 2000                     # server_app.py:155 FixedEvalManager(samples_per_client=2000)
HOLDOUT_SEED = 42                      # FixedEvalManager default seed (no seed passed)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "20260727", "holdout_overlap_audit",
)


def _binarize_like_fixed_eval(labels: np.ndarray) -> np.ndarray:
    """Mirror rmc/fixed_eval.py:64-67 label handling."""
    if labels.max() > 1:
        return (labels > 0).astype(np.int64)
    return labels.astype(np.int64)


def reconstruct_holdout_indices():
    """Reconstruct per-client holdout original-row positions exactly as
    rmc/fixed_eval.py::_build_holdout draws them (shared RandomState in file order)."""
    cfg = get_dataset_config(EVAL_DATASET)
    data_dir = cfg["data_dir"]
    label_col = cfg["label_column"]
    client_files = cfg["client_files"]

    rng = np.random.RandomState(HOLDOUT_SEED)  # SINGLE shared rng, file order (fixed_eval.py:47)
    per_client = {}
    total_selected = 0
    total_pos = 0

    for fname in client_files:
        client_idx = int(fname.split("_")[1].split(".")[0])
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[holdout] WARNING missing {path}; skipping (matches fixed_eval)")
            continue
        labels_raw = pd.read_parquet(path, columns=[label_col])[label_col].values.copy()
        labels = _binarize_like_fixed_eval(labels_raw)
        n = min(HOLDOUT_SPC, len(labels))
        if n < len(labels):
            unique_labels = np.unique(labels)  # sorted ascending, matches np.unique in fixed_eval
            indices = []
            for lbl in unique_labels:
                lbl_indices = np.where(labels == lbl)[0]
                proportion = len(lbl_indices) / len(labels)
                n_sample = max(1, int(n * proportion))
                if n_sample > len(lbl_indices):
                    n_sample = len(lbl_indices)
                sampled = rng.choice(lbl_indices, size=n_sample, replace=False)
                indices.extend(sampled.tolist())
            indices = np.array(indices, dtype=np.int64)
        else:
            indices = np.arange(len(labels), dtype=np.int64)

        sel = set(int(i) for i in indices)
        per_client[client_idx] = {
            "indices": sel,
            "n_holdout": len(sel),
            "n_holdout_pos": int(labels[indices].sum()),
        }
        total_selected += len(indices)
        total_pos += int(labels[indices].sum())
        del labels_raw, labels
        gc.collect()
        print(f"[holdout] client_{client_idx}: {len(sel)} rows selected")

    return per_client, total_selected, total_pos, label_col


def reconstruct_train_indices(
    client_idx: int,
    data_dir: str,
    label_col: str,
    max_samples: int = MAX_SAMPLES_PER_CLIENT,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    fname: "str | None" = None,
    dataset_name: str = TRAIN_DATASET,
):
    """Reconstruct the TRAIN-split original-row positions for one partition,
    exactly as flowerfl/task.py::load_data produces them at the given cap.

    The keyword args default to this module's audited constants (2M cap, 0.8/0.1
    split, TRAIN_DATASET) so the audit's own call sites are byte-identical;
    rmc/fixed_eval.py passes the ACTIVE run's cap/split/dataset when it reuses
    this reconstruction to build a disjoint holdout (GWU-61), so the excluded
    rows match what load_data actually routed into training for THAT run.

    ``client_idx`` is the ORDINAL position of the partition in the config's
    client_files list (NOT a numeric id parsed out of the filename) — it is the
    exact quantity load_data uses for the cap's ``random_state = 42 + client_idx``. ``fname`` is the actual parquet filename to read; when
    None it falls back to ``client_{client_idx}.parquet`` (the canonical
    edge_full_20 layout, where ordinal == numeric id).

    ``dataset_name`` is the TRAINING dataset name; the stratified cap is applied
    if and only if load_data would apply it (``is_full_dataset``), NOT
    unconditionally on ``n_rows > max_samples``.

    Returns (train_orig_positions:set, meta:dict)."""
    path = os.path.join(data_dir, fname if fname is not None else f"client_{client_idx}.parquet")
    labels = pd.read_parquet(path, columns=[label_col])[label_col].values
    n_rows = len(labels)

    # --- stratified cap (task.py) — only for is_full_dataset AND len(df) > cap,
    # mirroring load_data's dataset-specific gate exactly (P1-4).
    capped = is_full_dataset(dataset_name) and n_rows > max_samples
    if capped:
        df = pd.DataFrame({label_col: labels, "_pos": np.arange(n_rows, dtype=np.int64)})
        sampled_parts = []
        for label_val, group in df.groupby(label_col):  # sort=True (default), matches task.py
            n_samples = max(1, int(max_samples * len(group) / len(df)))
            sampled_parts.append(group.sample(n=n_samples, random_state=42 + client_idx))
        capped_df = pd.concat(sampled_parts).reset_index(drop=True)
        capped_orig_positions = capped_df["_pos"].to_numpy()  # frame-row -> original parquet row
        total = len(capped_df)
        del df, sampled_parts, capped_df
    else:
        capped_orig_positions = None  # frame row == original row
        total = n_rows

    # --- random_split (task.py:698-702) --------------------------------------
    train_size = int(train_split * total)
    val_size = int(val_split * total)
    # torch.random_split: indices = randperm(total, generator=g); train = indices[:train_size]
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(total, generator=g).numpy()
    train_frame_pos = perm[:train_size]
    val_frame_pos = perm[train_size:train_size + val_size]

    if capped:
        train_orig = capped_orig_positions[train_frame_pos]
        val_orig = capped_orig_positions[val_frame_pos]
        in_cap = set(int(p) for p in capped_orig_positions)
    else:
        train_orig = train_frame_pos
        val_orig = val_frame_pos
        in_cap = None

    meta = {
        "n_rows_full": int(n_rows),
        "capped": bool(capped),
        "n_after_cap": int(total),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "cap_dropped_rows": int(n_rows - total) if capped else 0,
    }
    result = {
        "train": set(int(p) for p in train_orig),
        "val": set(int(p) for p in val_orig),
        "in_cap": in_cap,
    }
    del labels
    gc.collect()
    return result, meta


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== holdout/train overlap audit @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"train={TRAIN_DATASET} eval={EVAL_DATASET} cap={MAX_SAMPLES_PER_CLIENT} "
          f"holdout_spc={HOLDOUT_SPC} seed={SEED}")

    holdout, holdout_total, holdout_pos, label_col = reconstruct_holdout_indices()

    # ---- fidelity check: real FixedEvalManager total + positive count --------
    fidelity = {"checked": False}
    try:
        from rmc.fixed_eval import FixedEvalManager
        print("[fidelity] constructing real FixedEvalManager (edge_full_20, spc=2000)...")
        fem = FixedEvalManager(dataset_name=EVAL_DATASET, samples_per_client=HOLDOUT_SPC, seed=HOLDOUT_SEED)
        ys = []
        for _, yb in fem._testloader:
            ys.append(yb.numpy())
        ys = np.concatenate(ys)
        real_total, real_pos = int(len(ys)), int(ys.sum())
        fidelity = {
            "checked": True,
            "reconstructed_total": holdout_total, "real_total": real_total,
            "reconstructed_pos": holdout_pos, "real_pos": real_pos,
            "total_match": holdout_total == real_total,
            "pos_match": holdout_pos == real_pos,
        }
        print(f"[fidelity] reconstructed total/pos = {holdout_total}/{holdout_pos}; "
              f"real = {real_total}/{real_pos}; "
              f"match={fidelity['total_match'] and fidelity['pos_match']}")
        del fem, ys
        gc.collect()
    except Exception as e:  # pragma: no cover
        fidelity = {"checked": False, "error": repr(e)}
        print(f"[fidelity] WARNING could not run real FixedEvalManager: {e!r}")

    # ---- per-client overlap --------------------------------------------------
    train_cfg = get_dataset_config(TRAIN_DATASET)
    data_dir = train_cfg["data_dir"]
    n_partitions = len(train_cfg["client_files"])

    per_client_rows = []
    agg_overlap_train = 0
    agg_holdout = 0
    for client_idx in sorted(holdout.keys()):  # holdout only covers eval files (0..19)
        H = holdout[client_idx]["indices"]
        tr, meta = reconstruct_train_indices(client_idx, data_dir, label_col)
        ov_train = len(H & tr["train"])
        ov_val = len(H & tr["val"])
        # holdout rows that survived the cap (only meaningful when capped)
        if tr["in_cap"] is not None:
            holdout_in_cap = len(H & tr["in_cap"])
        else:
            holdout_in_cap = len(H)  # uncapped: entire file present, all holdout rows in the pool
        row = {
            "client": client_idx,
            "n_holdout": len(H),
            "n_train": len(tr["train"]),
            "overlap_train": ov_train,
            "overlap_val": ov_val,
            "pct_holdout_in_train": round(100.0 * ov_train / len(H), 4) if H else 0.0,
            "pct_train_in_holdout": round(100.0 * ov_train / len(tr["train"]), 6) if tr["train"] else 0.0,
            "holdout_rows_surviving_cap": holdout_in_cap,
            **meta,
        }
        per_client_rows.append(row)
        agg_overlap_train += ov_train
        agg_holdout += len(H)
        print(f"[overlap] client_{client_idx}: holdout={len(H)} train={len(tr['train'])} "
              f"overlap_train={ov_train} ({row['pct_holdout_in_train']}%)  capped={meta['capped']}")
        del H, tr
        gc.collect()

    overall_pct = round(100.0 * agg_overlap_train / agg_holdout, 4) if agg_holdout else 0.0

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_dataset": TRAIN_DATASET, "eval_dataset": EVAL_DATASET,
            "seed": SEED, "max_per_client": MAX_SAMPLES_PER_CLIENT,
            "holdout_samples_per_client": HOLDOUT_SPC, "holdout_seed": HOLDOUT_SEED,
            "train_split": TRAIN_SPLIT, "val_split": VAL_SPLIT,
            "n_train_partitions": n_partitions,
            "n_eval_files_in_holdout": len(holdout),
        },
        "fidelity_check": fidelity,
        "aggregate": {
            "total_holdout_rows": agg_holdout,
            "total_holdout_rows_also_in_train": agg_overlap_train,
            "pct_holdout_contaminated_overall": overall_pct,
        },
        "per_client": per_client_rows,
        "notes": [
            "Holdout is built from eval_dataset=edge_full_20 (client_0..client_19); "
            "train partition client_20 exists in edge_full_20_rmc but is never in the holdout.",
            "Row selection depends only on labels+counts, so only the label column was read; "
            "identical rows are selected as the full-column load_data path.",
            "Train path is a line-faithful re-implementation (load_data returns loaders, not "
            "indices); holdout path is validated against the real FixedEvalManager totals.",
        ],
    }

    out_json = os.path.join(OUT_DIR, "overlap.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {out_json}")
    print(f"[done] OVERALL holdout contamination: {agg_overlap_train}/{agg_holdout} = {overall_pct}%")
    return result


if __name__ == "__main__":
    main()
