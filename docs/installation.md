# Installation

This project is distributed as a source checkout. Python 3.10 is required; the
package metadata deliberately excludes Python 3.11 and later.

```bash
git clone https://github.com/erikdj/federated-learning-rmc-defense.git
cd federated-learning-rmc-defense
git switch -c my-experiments v0.1.1
```

## Reference environment

[`environment-aws.yml`](../environment-aws.yml) is the pinned CPU reference
environment used by the released runtime. It currently specifies Python
3.10.20, Flower 1.29.0, PyTorch 2.11.0+cpu, NumPy 2.2.6, pandas 2.3.3,
PyArrow 23.0.1, and scikit-learn 1.7.2, together with the rest of the resolved
dependency set. Its system packages target 64-bit Linux.

From the repository root, create the environment with Conda:

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
  conda env create -f environment-aws.yml
conda activate flowerfl
python -m pip install -e .
```

The extra index supplies the pinned CPU-only PyTorch wheel, matching the
released container build.

The final command installs the current checkout in editable mode, including
the `flowerfl`, `rmc`, and `praxis_exp` packages and the `praxis` command.
Installing the package is necessary even after creating the Conda environment:
the environment file installs dependencies, while the editable install makes
this source tree importable.

Verify the interpreter and core imports:

```bash
python --version
python -c "import flwr, numpy, pandas, pyarrow, sklearn, torch; print(flwr.__version__, torch.__version__)"
python -c "import flowerfl, praxis_exp, rmc; print('source checkout imports OK')"
praxis --help
```

`python --version` should report `Python 3.10.20` in the reference environment.

## Portable development environment

On a platform where the Linux Conda lock cannot be solved, use any Python 3.10
interpreter and install the declared dependency ranges from `pyproject.toml`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This is suitable for development and offline verification, but it is not the
exact pinned reference environment. [`requirements.txt`](../requirements.txt)
is the pip-form CPU reference extracted from the Conda environment and is an
alternative when an exact pip dependency set is needed:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Verification

The default pytest configuration excludes tests marked `slow`, `anchor`, `ray`,
and `golden`. Run the software suite with:

```bash
python -m pytest -q
```

Tests that launch Ray, full-data anchors, AWS deployments, and experiment
fleets have separate prerequisites and are intentionally outside this basic
installation check. Dataset preprocessing and ordinary CPU simulations do not
require a GPU. Full-data preparation is memory and disk intensive; see
[Data and split compatibility](data.md) before downloading or generating data.

### Strict H2′ numerical gate

Before scoring any H2′ confirmatory corpus, run:

```bash
python -m pytest -q -m golden
```

Both marked checks compare the original canonical floating-point output byte for
byte. They fail on the public-release preparation host because derived slopes
differ in their final floating-point digits. Changing OpenBLAS CPU targets did
not reproduce the frozen hash. A fresh Python 3.10.20 installation of the complete
pip reference also reproduces this mismatch on that host. A compatible environment must pass this gate;
package pins alone are not proof of compatibility. The adjudicator enforces the
same gate unconditionally. Do not edit the golden values or bypass it.

The published evidence can be inspected and its arithmetic verified without
rerunning that raw-corpus adjudication. The separately dispatched GitHub protocol
workflow reports the strict gate independently from the software test suite.
