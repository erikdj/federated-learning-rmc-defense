# Contributing

Open an issue describing the behavior or experiment you want to change. Include a minimal example, the exact commit, Python/package versions, and the relevant test or artifact evidence.

Use a source checkout and the installation guide. Run the offline tests for the affected components. AWS tests should use fake services; do not make tests depend on personal credentials or live resources. Never commit credentials, downloaded raw datasets, local experiment outputs, or manuscript material.

Keep research instruments stable. A changed feature builder, seed set, threshold, split, model, attack schedule, or aggregation order defines a new experiment version. Add a new configuration and record the change instead of silently replacing a frozen instrument. Preserve negative findings and distinguish confirmatory tests from diagnostic reruns.

New harness behavior should have a regression test covering its failure path. Changes to S3 commit ordering, retries, identity of MLflow runs, or refill selection need explicit review of duplicate and interrupted execution. Do not claim exactly-once execution or checkpoint recovery unless new code actually establishes it.

The `praxis` CLI requires a Git source checkout. Version experiment documents before launching. Review the fully expanded matrix and resource settings before starting a fleet, and retain its manifest with the resulting artifacts.

The release manifest covers all tracked files. After reviewing a change and staging
its intended files, refresh the payload checksums and run the verifier:

```bash
python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
path = Path("RELEASE_MANIFEST.json")
manifest = json.loads(path.read_text())
names = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
manifest["files"] = [
    {"path": name, "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest()}
    for name in sorted(names) if name and name != path.name
]
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
python scripts/release/verify.py
git add RELEASE_MANIFEST.json
```

This updates the software payload inventory. The separate evidence and protocol
checksums describe frozen research records; do not refresh them to hide a changed
instrument or a failing numerical gate.
