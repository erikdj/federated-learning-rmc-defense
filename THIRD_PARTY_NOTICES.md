# Third-party research and software

## Dataset

The optional processed Edge-IIoTset download is adapted dataset material under
CC BY 4.0, separately from this repository's MIT software license. Attribution,
the dataset DOI, transformation details and notices are in
[data/DATASET_LICENSE.md](data/DATASET_LICENSE.md).

## Research foundations

The reconnecting-client threat model and baseline experimental architecture
build on Szelag, Chin, Ansell and Yip, *Integrating Identity-Based Identification
against Adaptive Adversaries in Federated Learning* ([paper](https://arxiv.org/abs/2504.03077),
[authors' implementation](https://github.com/SzelagJK/IBFL)). Cite the original
work when using those concepts. Historical standalone scripts containing copied
upstream baseline implementations are excluded from this public release; the
published authors' repository is the entry point for that separate baseline.

Krum, federated averaging, isolation forests, gradient-boosted trees and LSTM
models retain their respective research provenance. This repository supplies
experiment orchestration, scenario integration and the released research
implementations; it does not claim authorship of those underlying algorithms.

## Python and infrastructure dependencies

Flower, PyTorch, NumPy, SciPy, pandas, scikit-learn, MLflow, Ray, boto3 and other
dependencies are distributed by their respective projects under their own
licenses. Dependency versions are recorded in `environment-aws.yml` and
`requirements.txt`; their licenses are not replaced by this repository's MIT
license. The container base and operating-system packages likewise retain their
upstream terms.
