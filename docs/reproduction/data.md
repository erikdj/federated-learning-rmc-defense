# Data preparation

For the closest historical reproduction, first use the checked processed-data
archive linked by the repository's [main data guide](../data.md) and verify its
published checksums. The raw preparation below is the fallback for building a
new corpus when that release artifact is unavailable.

The experiments use the full Edge-IIoTset normal-traffic and attack-traffic CSV
collection, with binary `Attack_label`. Obtain the dataset from the creator's
[IEEE DataPort record](https://doi.org/10.21227/mbc1-1h68) and cite the accompanying
[IEEE Access article](https://doi.org/10.1109/ACCESS.2022.3165809). Confirm the
dataset version and access terms shown by the source record when downloading.
The data files are not covered by this repository's MIT license.

Arrange the extracted archive so one directory contains `Normal traffic/` and
`Attack traffic/`, then run:

```bash
conda run -n flowerfl python reproduction/prepare_edge_data.py \
  --dataset-root /path/to/Edge-IIoTset \
  --output-dir data/edge_full

conda run -n flowerfl python scripts/repartition_edge_20.py \
  --input-dir data/edge_full \
  --output-dir data/edge_full_20 \
  --alpha 0.5 \
  --seed 42
```

The first stage reads ten sensor-specific normal CSVs and fourteen attack CSVs,
drops identifiers, timestamps, string fields and `Attack_type`, converts the
remaining features to numeric values, replaces non-finite values with zero, and
produces ten Parquet clients. The second stage pools those clients and creates
20 binary-label non-IID partitions using class-wise Dirichlet allocation with
α = 0.5 and seed 42. It copies `client_19.parquet` byte-for-byte to
`client_20.parquet` for the reconnecting identity.

The experiment dataset name is `edge_full_20_rmc`. Training exposes 21 logical
partitions, while server evaluation maps that name to `data/edge_full_20`.
Keep the committed [`holdout manifest`](../../data/holdout_manifest_edge_full_20_rmc.json)
and [`validation/test split`](../../data/val_test_split_manifest.json) unchanged.
The holdout samples 2,000 rows from each of the 21 partitions with seed 20260416;
the subsequent per-partition, per-label split uses validation fraction 0.7 and
seed 20260515. H4 and the proposed H4′ use the sealed-test side of that split.

The raw source-archive digest was not recorded in the research repository. The
holdout manifest records row indices and truncated per-partition source hashes,
but that does not identify the original archive. Consequently, the raw commands
above provide the recorded transformation and an internally reproducible fresh
corpus; they cannot establish that a later download is byte-identical to the
corpus used for the published numerical results. Use the separately released
processed archive and its full content hashes for byte-exact historical input.
