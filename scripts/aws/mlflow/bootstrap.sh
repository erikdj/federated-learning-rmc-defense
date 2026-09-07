#!/usr/bin/env bash
# Run as root on a fresh Amazon Linux 2023 MLflow host after creating
# /etc/fl-rmc-mlflow.env. The file contains resource identifiers, not a DB password.
set -euo pipefail
[[ "${EUID:-$(id -u)}" -eq 0 ]] || { echo "ERROR: run as root" >&2; exit 1; }
[[ -f /etc/fl-rmc-mlflow.env ]] || { echo "ERROR: missing /etc/fl-rmc-mlflow.env" >&2; exit 1; }

dnf -y install awscli jq python3.11
python3.11 -m venv /opt/mlflow
/opt/mlflow/bin/pip install --upgrade pip
/opt/mlflow/bin/pip install 'mlflow==3.14.0' boto3 gunicorn psycopg2-binary

install -m 0755 scripts/aws/mlflow/start_mlflow.sh /usr/local/bin/fl-rmc-start-mlflow
install -m 0644 scripts/aws/mlflow/fl-rmc-mlflow.service /etc/systemd/system/fl-rmc-mlflow.service
systemctl daemon-reload
systemctl enable --now fl-rmc-mlflow.service

