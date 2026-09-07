# docker/Dockerfile — praxis-fleet "gold" image (SP3).
# Built + pushed by scripts/aws/ecr/build_push.sh, tagged with the repo short-SHA,
# referenced downstream by immutable digest (image@sha256) as the chain-of-custody anchor.
#
# Reproducibility notes:
#  - environment-aws.yml is the canonical, fully build-pinned env (kept byte-identical).
#  - The dataset is NOT baked: data/edge_full_20/ is synced from S3 at container start.
#  - PIP_EXTRA_INDEX_URL supplies the PyTorch CPU wheel index so the pinned
#    `torch==2.11.0+cpu` resolves (it is not on PyPI). This is build mechanics only —
#    it changes no pinned version, so it is not a methodology change.
FROM mambaorg/micromamba:1.5-bookworm-slim
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git awscli \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Create the locked env. PyTorch CPU wheels live on download.pytorch.org, not PyPI.
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY environment-aws.yml /tmp/environment-aws.yml
RUN micromamba create -y -n flowerfl -f /tmp/environment-aws.yml && micromamba clean -ay

# Bake the repo (build context filtered by .dockerignore: no data/logs/results/.git).
COPY . /app
# Runtime provenance: the repo SHA being baked, passed by
# build_push.sh via --build-arg. The ENV makes it the runner's own commit —
# .git is filtered out so run_phase4_flower.py cannot resolve it otherwise.
ARG PRAXIS_RUNNER_COMMIT=unknown
ENV PRAXIS_RUNNER_COMMIT=$PRAXIS_RUNNER_COMMIT
ENV PATH=/opt/conda/envs/flowerfl/bin:$PATH PYTHONPATH=/app

# Stage the dataset from S3, then resolve this array index's unit and run it.
# NON-login shell (-c, not -lc) + absolute python: the micromamba base image's
# login-shell profile RESETS PATH, dropping the conda env — every Batch task
# died with exit 127 "python: command not found" (2026-07-12 launch #2; the
# import smoke missed it because --entrypoint bypasses the login shell).
ENTRYPOINT ["/bin/bash","-c","aws s3 sync s3://$PRAXIS_BUCKET/data/edge_full_20/ data/edge_full_20/ --only-show-errors && /opt/conda/envs/flowerfl/bin/python docker/entrypoint.py"]
