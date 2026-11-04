"""Train the compact variable-size Random Forest for the optimisation trial.

When the standalone training CSVs are absent, regenerate the canonical N=4..10
training/validation split with the repository's dataset generator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from trial import ROOT, OUT, train_model

PROJECT_ROOT = ROOT.parents[1]
PACKAGE_DATA = ROOT / "data"
PROJECT_DATA = PROJECT_ROOT / "data"


def main() -> None:
    if (PACKAGE_DATA / "train.csv").is_file() and (PACKAGE_DATA / "validation.csv").is_file():
        data_dir = PACKAGE_DATA
    else:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.dataset_generator import generate_dataset

        data_dir = PROJECT_DATA
        if not (data_dir / "train.csv").is_file() or not (data_dir / "validation.csv").is_file():
            data_dir.mkdir(parents=True, exist_ok=True)
            generate_dataset(
                board_sizes=(4, 5, 6, 7, 8, 9, 10),
                csv_path=data_dir / "training_data.csv",
                summary_path=data_dir / "dataset_summary.json",
                train_path=data_dir / "train.csv",
                validation_path=data_dir / "validation.csv",
            )

    _, metrics, _ = train_model(data_dir, OUT)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
