from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / CONFIG["dataset"]["file"]

if not DATA.exists():
    raise SystemExit(f"Dataset not found: {DATA}")

h = hashlib.sha256()
with DATA.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(block)

actual_hash = h.hexdigest()
columns = pd.read_csv(DATA, nrows=0).shape[1]
rows = sum(1 for _ in DATA.open("rb")) - 1

print(f"Rows: {rows:,}")
print(f"Columns: {columns}")
print(f"SHA-256: {actual_hash}")

expected = CONFIG["dataset"]
if rows != expected["expected_rows"]:
    raise SystemExit("Row count does not match the audited dataset.")
if columns != expected["expected_columns"]:
    raise SystemExit("Column count does not match the audited dataset.")
if actual_hash != expected["expected_sha256"]:
    raise SystemExit("SHA-256 does not match the audited dataset version.")

print("PASS: dataset matches the version used in the paper.")
