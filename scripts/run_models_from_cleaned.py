"""Reproduce model, SHAP, ranking, and figure outputs from included cleaned data."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

import build_cleaned_and_analyze as ra

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "cleaned"
INVENTORY_PATH = REPO_ROOT / "data" / "metadata" / "data_inventory.csv"


def load_cleaned_datasets():
    inventory = pd.read_csv(INVENTORY_PATH)
    datasets = []
    for row in inventory.itertuples(index=False):
        folder = DATA_ROOT / row.site / f"segment{int(row.segment)}"
        individual_raw = pd.read_csv(folder / "individual_features_raw_150.csv")
        comprehensive_raw = pd.read_csv(folder / "comprehensive_features_raw_150.csv")
        individual = pd.read_csv(folder / "individual_features_smoothed_150.csv")
        comprehensive = pd.read_csv(folder / "comprehensive_features_smoothed_150.csv")
        trajectory_raw = pd.read_csv(folder / "trajectory_cleaned_150.csv")
        trajectory = pd.read_csv(folder / "trajectory_smoothed_150.csv")
        datasets.append(ra.Dataset(
            site=row.site,
            segment=int(row.segment),
            frame=individual["Frame"].to_numpy(float),
            individual_raw=individual_raw,
            comprehensive_raw=comprehensive_raw,
            individual=individual,
            comprehensive=comprehensive,
            trajectory_raw=trajectory_raw,
            trajectory=trajectory,
            raw_points=int(row.pedestrian_raw_points),
            trajectory_ids=int(row.trajectory_ids),
            velocity_removed=int(row.velocity_values_removed_above_2m_s),
        ))
    return datasets, inventory


def main():
    ra.RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    ra.MODELS_OUT.mkdir(parents=True, exist_ok=True)
    ra.FIG_OUT.mkdir(parents=True, exist_ok=True)
    datasets, inventory = load_cleaned_datasets()
    descriptive = ra.descriptive_tables(datasets)
    # Keep metadata in the data directory as the repository's source of truth.
    descriptive.to_csv(REPO_ROOT / "data" / "metadata" / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    performance, _, importance = ra.run_models(datasets)
    table5, table6 = ra.aggregate_rankings(importance)
    ra.create_figures(importance)
    summary = ra.build_summary(inventory, performance, descriptive, table5, table6)
    print(json.dumps({
        "datasets": len(datasets),
        "models": len(performance),
        "figures": summary["figures"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
