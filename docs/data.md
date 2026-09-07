# Data provenance and split compatibility

The raw Edge-IIoTset files and generated Parquet partitions are not distributed
in this source tree. The repository contains preprocessing code and frozen
index manifests. Edge-IIoTset and material derived from it remain subject to
the dataset's license and attribution requirements.

## Upstream source and rights

Edge-IIoTset was created by Mohamed A. Ferrag, Othmane Friha, Djallel Hamouda,
Leandros Maglaras, and Helge Janicke. Obtain it from the authors' official
[IEEE DataPort record](https://doi.org/10.21227/mbc1-1h68) and cite the
[associated IEEE Access paper](https://doi.org/10.1109/ACCESS.2022.3165809).
The [Edith Cowan University publication record](https://ro.ecu.edu.au/ecuworks2022-2026/552/)
also links the paper and dataset records.

The [DataCite metadata for the dataset DOI](https://api.datacite.org/dois/10.21227/mbc1-1h68)
names the dataset license as
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)
(`CC-BY-4.0`). The DataPort page also grants academic research use indefinitely
and asks commercial users to obtain permission from the lead author, Dr Mohamed
Amine Ferrag. These notices express different commercial-use conditions; this
release discloses both and does not claim that their relationship has been
resolved or that separate commercial permission has been obtained. Consult the
author through the upstream record for commercial-use clarification. The
upstream commercial-use wording and verification date are preserved in
[`data/DATASET_LICENSE.md`](../data/DATASET_LICENSE.md).

The repository's MIT license applies to its software. It is not a replacement
for the dataset license. The processed Parquet release identifies itself as
adapted Edge-IIoTset material, credits Ferrag et al., cites the dataset DOI,
links CC BY 4.0, and summarizes the transformations below. Its complete notice
is in [`data/DATASET_LICENSE.md`](../data/DATASET_LICENSE.md) and inside the
archive.

## Recorded data lineage

The upstream corpus is reported as 20,952,648 records and contains traffic from
more than ten IoT/IIoT device types and fourteen attacks. The final project data
artifact is a later processing stage:

| Stage | Rows represented | Features | Partitioning |
|---|---:|---:|---|
| Published Edge-IIoTset corpus | 20,952,648 | 1,176 collected; 61 high-correlation features described by the authors | Upstream package |
| `data/edge_full` project artifact | 20,939,617 | 45 numeric protocol features plus binary `Attack_label` | 10 sensor clients |
| `data/edge_full_20` project artifact | 20,939,617 unique rows | Same 45 features plus label | 20 Dirichlet partitions; `alpha=0.5`, seed 42 |
| `client_20.parquet` | 207,144 additional stored rows | Same schema | Retained byte copy of `client_19.parquet`; not the scenario ladder's reconnect mechanism |

The two 20.9-million counts refer to different stages and must not be treated
as a simple raw-minus-dropped-rows calculation. `process_edge_full.py` samples
attack traffic independently for each sensor client with seeds 42 through 51,
uses integer proportional allocations, and does not retain upstream row IDs.
Although an old source comment says “without replacement across clients,” the
implementation does not maintain a shared exclusion set between those samples.

The main pipeline performs these transformations:

1. It reads ten normal sensor CSVs and every CSV in `Attack traffic` (the
   expected upstream layout has fourteen attack CSVs).
2. It forces normal labels to 0 and attack labels to 1.
3. It drops timestamps, host identifiers, `Attack_type`, and thirteen string
   protocol fields; converts the remaining common sensor columns to numeric;
   and replaces infinities and missing values with zero.
4. It distributes sampled attack rows in proportion to each normal sensor's
   size, shuffles each client, and writes ten Parquet files.
5. `repartition_edge_20.py` concatenates those ten files and allocates each
   binary class independently across twenty clients with a Dirichlet draw
   (`alpha=0.5`, seed 42). Dataset metadata retains an earlier role assignment
   and a copy of client 19 as client 20. Executed S0–S4 scenarios assign the
   nine malicious base clients to slots 0–8. Their reconnect identities map
   back to the same original partition through `ScenarioStrategy`; they do not
   switch to the client-20 copy.

The output metadata's `total_rows` excludes the deliberate client-20 duplicate.
Counting all 21 stored Parquet files therefore produces 21,146,761 rows.

The excluded legacy `process_edge_full_encoded.py` comparison pipeline globally
label-encodes thirteen string protocol columns and emits 58 features under
`data/edge_full_enc`. It is not the 45-feature lineage used for the final study.
The earlier flat `data/edge/ML-EdgeIIoT-dataset.csv` path is also not an input to
the full-data scripts described here.

## Download the exact processed inputs

Release `v0.1.1` includes the exact processed partitions used by the frozen
row-index manifests. The archive is 289,809,085 bytes and its SHA-256 is
`f2b0e2a6aa3a72d6bbe6e01108bf22b7e8170fc6fd19680888d68fac3b6faa4d`.
[`data/processed-data-manifest.json`](../data/processed-data-manifest.json)
records the archive checksum and full SHA-256, byte size, Parquet row count,
and frozen fingerprint for every member.

From the repository root, download the asset with the GitHub CLI and verify it
before extraction:

```bash
mkdir -p .release-download
gh release download v0.1.1 \
  --repo erikdj/federated-learning-rmc-defense \
  --pattern edge-iiot-rmc-inputs-v2.tar.gz \
  --dir .release-download

DATA_ASSET=.release-download/edge-iiot-rmc-inputs-v2.tar.gz
EXPECTED_SHA=$(python -c \
  'import json; print(json.load(open("data/processed-data-manifest.json"))["release"]["sha256"])')
echo "$EXPECTED_SHA  $DATA_ASSET" | sha256sum --check --strict
if [ -e data/edge_full_20 ]; then
  echo "data/edge_full_20 already exists; refusing to replace it" >&2
  exit 1
fi
tar -xzf "$DATA_ASSET" -C .
```

The `test` command prevents accidental replacement of an existing dataset
directory. The archive extracts directly to `data/edge_full_20`, the path used
by `edge_full_20` and `edge_full_20_rmc` in `flowerfl.task.DATASET_CONFIGS`.

Verify every extracted member against its full public checksum:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("data/processed-data-manifest.json").read_text())
problems = []
for record in manifest["files"]:
    path = Path(record["path"])
    if not path.is_file():
        problems.append(f"missing: {path}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        problems.append(f"SHA-256 mismatch: {path}")
    if path.stat().st_size != record["byte_size"]:
        problems.append(f"byte-size mismatch: {path}")
if problems:
    raise SystemExit("\n".join(problems))
print("all processed-data files match the public manifest")
PY
```

This is the compatibility path for replaying the frozen index manifests. Raw
acquisition and preparation below create an independent dataset lineage.

## Acquire and check the raw package

Download and extract the official DataPort package. DataPort may require an
account and acceptance of its current terms. Because package names can change,
locate the extracted directory that directly contains both `Normal traffic`
and `Attack traffic`; do not substitute one of the flat ML/DNN CSV exports.

Set that directory as the raw-data root:

```bash
export EDGE_IIOT_RAW_DIR="/absolute/path/to/extracted/Edge-IIoTset/root"
```

Check the layout before starting the long conversion:

```bash
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["EDGE_IIOT_RAW_DIR"])
sensors = [
    "Distance", "Flame_Sensor", "Heart_Rate", "IR_Receiver", "Modbus",
    "phValue", "Soil_Moisture", "Sound_Sensor",
    "Temperature_and_Humidity", "Water_Level",
]
missing = [
    str(root / "Normal traffic" / sensor / f"{sensor}.csv")
    for sensor in sensors
    if not (root / "Normal traffic" / sensor / f"{sensor}.csv").is_file()
]
attacks = sorted((root / "Attack traffic").glob("*.csv"))
if missing or len(attacks) != 14:
    raise SystemExit(
        f"layout mismatch: missing normal CSVs={missing}; attack CSVs={len(attacks)}"
    )
print(f"layout OK: {len(sensors)} normal sensor CSVs, {len(attacks)} attack CSVs")
PY
```

## Build a new local partition set

Run these commands from a source checkout after installing the reference
environment:

```bash
conda activate flowerfl
python scripts/process_edge_full.py
python scripts/repartition_edge_20.py \
  --input-dir data/edge_full \
  --output-dir data/edge_full_20 \
  --alpha 0.5 \
  --seed 42
```

The first script writes `data/edge_full/client_0.parquet` through
`client_9.parquet`, `metadata.json`, and `features.json`. The second writes
twenty unique partitions, the client-20 identity copy, and its own metadata and
feature list. Both output directories are ignored by Git.

These programs use pandas batch operations. The first holds all attack rows
plus the ten cleaned normal frames in memory, and the repartitioner loads and
concatenates all ten generated Parquet files before splitting them. Use a
64-bit machine with enough RAM for multiple in-memory copies of roughly 20.9
million rows and disk space materially larger than the compressed download.
Data preparation is CPU based and does not require a GPU. No cloud service is
required, and resource cost depends on the machine chosen.

## Frozen study compatibility

The checked-in manifests are frozen research instruments:

- `data/holdout_manifest_edge_full_20_rmc.json` records 42,000 row positions
  (2,000 from each of 21 partitions), a sample of each source file's first
  megabyte hashed with SHA-256 and truncated to 16 hex characters, and sampling
  seed 20260416.
- `data/val_test_split_manifest.json` divides those positions into a fixed
  70/30 validation/test split with seed 20260515.

Those positions only mean the same thing against the original Parquet content
and row order. Reprocessing an upstream download with the same seeds does not
establish compatibility: package contents or CSV ordering may differ, and
library versions and Parquet serialization can change bytes. Exact frozen-index
compatibility requires the author-preserved `data/edge_full_20` bundle and
matching row counts and recorded drift fingerprints as minimum checks. The
fingerprints cover only the first megabyte of each file, so they are not
full-file authentication. The release asset adds full SHA-256 checksums and
ships the original Parquet and feature bytes. The only changed dataset file is
`metadata.json`, where one private local source path was replaced with
`data/edge_full`; its source and released hashes are both recorded. Use the
download and verification procedure above for an exact frozen-index replay.
Rebuilding from upstream raw data alone does not provide that compatibility.

Never regenerate either checked-in manifest in place. For an independently
prepared dataset, create visibly separate manifests:

```bash
python scripts/prepare_holdout_set.py \
  --dataset edge_full_20_rmc \
  --samples-per-partition 2000 \
  --seed 20260416 \
  --output data/holdout_manifest_edge_full_20_rmc.local.json

python scripts/data/split_holdout.py \
  --holdout-manifest data/holdout_manifest_edge_full_20_rmc.local.json \
  --val-frac 0.7 \
  --seed 20260515 \
  --out data/val_test_split_manifest.local.json
```

The preparation command resolves the `edge_full_20_rmc` dataset to
`data/edge_full_20`, so place only the new local Parquets there when running
it. If you possess the original bundle, preserve it elsewhere and work in a
separate checkout. Point new experiment and scoring configurations at the
`.local.json` files and give the data source a new identifier. Do not replace
the frozen filenames or describe results from the local rebuild as an exact
reproduction of the frozen study.

## Train/evaluation row overlap

A later audit of the canonical development honest-control run reconstructed a
39,980-row runtime evaluation set and found that 30,987 rows (77.5063%) also
appeared in some client's training split. This runtime evaluation set is
distinct from the checked-in 42,000-index manifest. The audit therefore does
not establish a row-disjoint H4 task evaluation, and absolute task-accuracy or
utility values from that regime should be interpreted with this limitation.
The audit does not alter detection metrics calculated from recorded update
signals, but the overlap remains relevant to utility comparisons based on model
accuracy.

For new experiments, construct train, validation, and test membership from a
single row-level allocation and exclude all evaluation indices, including the
source partition behind an RMC duplicate, from training. Record new manifests,
data hashes, software versions, and source-package provenance with those runs.
