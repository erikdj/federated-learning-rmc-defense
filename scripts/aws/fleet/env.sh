#!/usr/bin/env bash
# Shared configuration loader. This file contains no account-specific defaults.

if [[ -z "${PRAXIS_AWS_CONFIG:-}" ]]; then
  PRAXIS_AWS_CONFIG=".env.aws"
fi

if [[ ! -f "$PRAXIS_AWS_CONFIG" ]]; then
  echo "ERROR: AWS config not found: $PRAXIS_AWS_CONFIG" >&2
  echo "Copy scripts/aws/config.example.env to .env.aws and fill its placeholders." >&2
  return 2 2>/dev/null || exit 2
fi

# shellcheck disable=SC1090
source "$PRAXIS_AWS_CONFIG"

require_vars() {
  local missing=() name value
  for name in "$@"; do
    value="${!name:-}"
    if [[ -z "$value" || "$value" == *REPLACE* || "$value" == "000000000000" ]]; then
      missing+=("$name")
    fi
  done
  if ((${#missing[@]})); then
    echo "ERROR: configure these values in $PRAXIS_AWS_CONFIG: ${missing[*]}" >&2
    return 2
  fi
}

aws_cli() {
  command aws --region "$AWS_REGION" "$@"
}

