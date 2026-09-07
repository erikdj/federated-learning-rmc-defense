# Edge-IIoTset processed-data license and attribution

This notice applies to the processed dataset release
`edge-iiot-rmc-inputs-v1.tar.gz`. The archive is adapted from Edge-IIoTset and
is distributed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/legalcode)
(`CC-BY-4.0`).

## Required attribution

Credit the original dataset authors:

> Mohamed A. Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, and
> Helge Janicke. *Edge-IIoTset: A New Comprehensive Realistic Cyber Security
> Dataset of IoT and IIoT Applications: Centralized and Federated Learning.*
> IEEE DataPort, 2022. https://doi.org/10.21227/mbc1-1h68

The DataCite record for that DOI identifies the resource as a dataset and names
CC BY 4.0 as its license. The associated article is:

> M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke,
> “Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT
> and IIoT Applications for Centralized and Federated Learning,” *IEEE
> Access*, vol. 10, pp. 40281–40306, 2022.
> https://doi.org/10.1109/ACCESS.2022.3165809

## Changes in this adapted release

The project preprocessing code converted the upstream traffic into a binary
classification dataset, retained 45 numeric protocol features, replaced
missing and infinite numeric values with zero, and distributed sampled attack
rows among ten sensor-based clients. It then repartitioned the processed rows
into twenty non-IID client partitions using a Dirichlet allocation with
`alpha=0.5` and seed 42. `client_20.parquet` is a byte-identical copy of
`client_19.parquet` used to represent an identity reset by the same simulated
device.

The public archive preserves every Parquet file and `features.json` byte for
byte. It changes only the private local source-path value in `metadata.json` to
`data/edge_full`. Both metadata hashes and the precise redaction are recorded
in `data/processed-data-manifest.json` in the source release.

The original authors and IEEE do not endorse this adaptation or its research
uses. The repository's MIT license applies to the software source; this notice
governs the processed dataset archive.
