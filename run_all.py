from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    (1, "Dataset audit, preprocessing, and feature selection", ROOT / "src" / "01_prepare_data.py"),
    (2, "Holdout model evaluation", ROOT / "src" / "02_holdout_models.py"),
    (3, "Fold-specific five-fold cross-validation", ROOT / "src" / "03_cross_validation.py"),
    (4, "Exact McNemar tests and Holm correction", ROOT / "src" / "04_statistical_tests.py"),
    (5, "Controlled timing benchmark", ROOT / "src" / "05_timing_benchmark.py"),
    (6, "Generate manuscript figures", ROOT / "src" / "06_generate_figures.py"),
    (7, "Generate manuscript tables", ROOT / "src" / "07_generate_tables.py"),
]


def run_script(script: Path, title: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    elapsed_minutes = (time.perf_counter() - started) / 60
    print(f"Completed in {elapsed_minutes:.2f} minutes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete 5G-NIDD paper pipeline.")
    parser.add_argument("--from-stage", type=int, default=1, choices=range(1, 8))
    parser.add_argument("--to-stage", type=int, default=7, choices=range(1, 8))
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete generated work and results directories before running.",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip stage 5 because timing is hardware-dependent.",
    )
    arguments = parser.parse_args()

    if arguments.from_stage > arguments.to_stage:
        parser.error("--from-stage must not be greater than --to-stage")

    if arguments.clean:
        for directory in (ROOT / "work", ROOT / "results"):
            if directory.exists():
                shutil.rmtree(directory)
                print(f"Removed generated directory: {directory.relative_to(ROOT)}")

    run_script(ROOT / "check_dataset.py", "Dataset identity check")

    for number, title, script in STAGES:
        if not arguments.from_stage <= number <= arguments.to_stage:
            continue
        if arguments.skip_timing and number == 5:
            print("\n=== Stage 5/7: Controlled timing benchmark ===")
            print("Skipped by --skip-timing.")
            continue
        run_script(script, f"Stage {number}/7: {title}")


if __name__ == "__main__":
    main()
