"""Check the curated repository without requiring the excluded raw source files."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    checked = 0
    for site, count in (("nanjingdonglu", 7), ("wujiaochang", 5)):
        for segment in range(1, count + 1):
            folder = ROOT / "data" / "cleaned" / site / f"segment{segment}"
            expected = {
                "scene_header_repaired_150.csv": 531,
                "individual_features_raw_150.csv": None,
                "individual_features_smoothed_150.csv": None,
                "comprehensive_features_raw_150.csv": 13,
                "comprehensive_features_smoothed_150.csv": 13,
                "trajectory_cleaned_150.csv": 5,
                "trajectory_smoothed_150.csv": 5,
            }
            for name, ncols in expected.items():
                path = folder / name
                if not path.exists():
                    errors.append(f"missing: {path.relative_to(ROOT)}")
                    continue
                df = pd.read_csv(path)
                checked += 1
                if len(df) != 150:
                    errors.append(f"expected 150 rows: {path.relative_to(ROOT)}")
                if ncols is not None and len(df.columns) != ncols:
                    errors.append(f"unexpected columns: {path.relative_to(ROOT)}")
                if df.isna().any().any():
                    errors.append(f"NaN present: {path.relative_to(ROOT)}")

    inventory = pd.read_csv(ROOT / "data" / "metadata" / "data_inventory.csv")
    if len(inventory) != 12 or int(inventory.analysis_rows.sum()) != 1800:
        errors.append("inventory does not describe 12 segments and 1,800 analysis rows")

    performance = pd.read_csv(ROOT / "results" / "summary" / "model_performance.csv")
    if len(performance) != 56:
        errors.append(f"expected 56 model rows, found {len(performance)}")
    if set(performance.cv_method.dropna().unique()) != {"random_5fold_seed42"}:
        errors.append("unexpected cross-validation method")

    model_files = list((ROOT / "results" / "models").rglob("*.csv"))
    if len(model_files) != 168:
        errors.append(f"expected 168 per-model CSV files, found {len(model_files)}")
    figure_files = list((ROOT / "figures").glob("figure1[0-3]*"))
    if len(figure_files) != 12:
        errors.append(f"expected 12 figure files, found {len(figure_files)}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "cleaned_files_checked": checked,
        "analysis_rows": int(inventory.analysis_rows.sum()),
        "model_rows": len(performance),
        "per_model_files": len(model_files),
        "figure_files": len(figure_files),
        "errors": errors,
    }
    output = ROOT / "results" / "summary" / "repository_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
