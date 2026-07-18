from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("DEPTH_SOURCE_ROOT", REPO_ROOT))
DATA_OUT = REPO_ROOT / "data" / "cleaned"
RESULTS_OUT = REPO_ROOT / "results" / "summary"
MODELS_OUT = REPO_ROOT / "results" / "models"
FIG_OUT = REPO_ROOT / "figures"

SITES = {
    "nanjingdonglu": {"label": "East Nanjing Road", "segments": 7},
    "wujiaochang": {"label": "Wujiaochang", "segments": 5},
}

TAGS = [
    "wall", "floor", "ceiling", "windowpane", "cabinet", "sidewalk",
    "person", "door", "table", "chair", "painting", "mirror", "rug",
    "seat", "fence", "desk", "lamp", "railing", "box", "column",
    "signboard", "counter", "grandstand", "path", "stairs", "case",
    "stairway", "toilet", "bench", "countertop", "computer", "bar",
    "light", "chandelier", "booth", "pole", "bannister", "escalator",
    "ottoman", "bottle", "buffet", "poster", "stage", "plaything",
    "stool", "barrel", "step", "screen door", "sconce", "ashcan",
    "monitor", "radiator", "glass",
]

OBSTACLE_TAGS = ["fence", "railing", "column", "bar", "bannister"]
VISIBILITY_TAGS = ["glass", "windowpane"]
STRUCTURAL_TAGS = ["wall", "floor", "ceiling", "stairs", "stairway"]

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

FEATURE_LABELS = {
    "near_ratio": "Proximal area ratio",
    "middle_ratio": "Medial area ratio",
    "far_ratio": "Distal area ratio",
    "obstacle_near_density": "Proximal obstacle density",
    "obstacle_middle_density": "Medial obstacle density",
    "obstacle_far_density": "Distal obstacle density",
    "visibility_near_ratio": "Proximal visibility ratio",
    "visibility_middle_ratio": "Medial visibility ratio",
    "visibility_far_ratio": "Distal visibility ratio",
    "clutter_near_degree": "Proximal clutter degree",
    "clutter_middle_degree": "Medial clutter degree",
    "clutter_far_degree": "Distal clutter degree",
}


@dataclass
class Dataset:
    site: str
    segment: int
    frame: np.ndarray
    individual_raw: pd.DataFrame
    comprehensive_raw: pd.DataFrame
    individual: pd.DataFrame
    comprehensive: pd.DataFrame
    trajectory_raw: pd.DataFrame
    trajectory: pd.DataFrame
    raw_points: int
    trajectory_ids: int
    velocity_removed: int


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Match the manuscript's existing centered moving-average implementation."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values.copy()
    smoothed = np.convolve(values, np.ones(window) / window, mode="same")
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


def read_scene_values(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read the first 541 real fields and reconstruct the ten 2-m depth bands."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 151:
        raise ValueError(f"{path} has only {len(rows)-1} data rows")

    original_header_fields = len(rows[0])
    parsed = []
    frames = []
    marker_errors = []
    for row_no, row in enumerate(rows[1:151], start=2):
        if len(row) < 541:
            raise ValueError(f"{path}, row {row_no}: expected >=541 fields, got {len(row)}")
        row = row[:541]
        frames.append(float(row[0]))
        levels = []
        for level in range(1, 11):
            start = 1 + (level - 1) * 54
            marker = row[start].strip()
            if marker != f"level{level}":
                marker_errors.append((row_no, level, marker))
            vals = pd.to_numeric(pd.Series(row[start + 1:start + 54]), errors="coerce").to_numpy(float)
            if np.isnan(vals).any():
                raise ValueError(f"{path}, row {row_no}, level {level}: nonnumeric scene value")
            levels.append(vals)
        parsed.append(levels)
    if marker_errors:
        raise ValueError(f"{path}: depth marker errors, first={marker_errors[0]}")
    meta = {
        "source_header_fields": original_header_fields,
        "discarded_header_fields": max(0, original_header_fields - 541),
        "rows_used": 150,
    }
    return np.asarray(frames), np.asarray(parsed), meta


def aggregate_scene(frames: np.ndarray, levels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # levels: observation x 10 depth levels x 53 semantic tags
    zones = {
        "near": levels[:, 0, :],
        "middle": levels[:, 1:5, :].sum(axis=1),
        "far": levels[:, 5:10, :].sum(axis=1),
    }
    individual = pd.DataFrame({"Frame": frames})
    for tag_idx, tag in enumerate(TAGS):
        for zone in ("near", "middle", "far"):
            individual[f"{tag}.{zone}"] = zones[zone][:, tag_idx]

    totals = {zone: arr.sum(axis=1) for zone, arr in zones.items()}
    all_total = totals["near"] + totals["middle"] + totals["far"]
    comprehensive = pd.DataFrame({"Frame": frames})
    for zone in ("near", "middle", "far"):
        comprehensive[f"{zone}_ratio"] = np.divide(
            totals[zone], all_total, out=np.zeros_like(all_total), where=all_total != 0
        )
    for zone in ("near", "middle", "far"):
        denom = totals[zone]
        obstacle = zones[zone][:, [TAGS.index(t) for t in OBSTACLE_TAGS]].sum(axis=1)
        visibility = zones[zone][:, [TAGS.index(t) for t in VISIBILITY_TAGS]].sum(axis=1)
        clutter_idx = [i for i, t in enumerate(TAGS) if t not in STRUCTURAL_TAGS]
        clutter = zones[zone][:, clutter_idx].sum(axis=1)
        comprehensive[f"obstacle_{zone}_density"] = np.divide(
            obstacle, denom, out=np.zeros_like(denom), where=denom != 0
        )
        comprehensive[f"visibility_{zone}_ratio"] = np.divide(
            visibility, denom, out=np.zeros_like(denom), where=denom != 0
        )
        comprehensive[f"clutter_{zone}_degree"] = np.divide(
            clutter, denom, out=np.zeros_like(denom), where=denom != 0
        )

    # Reorder to the twelve-variable manuscript order.
    comprehensive = comprehensive[[
        "Frame", "near_ratio", "middle_ratio", "far_ratio",
        "obstacle_near_density", "obstacle_middle_density", "obstacle_far_density",
        "visibility_near_ratio", "visibility_middle_ratio", "visibility_far_ratio",
        "clutter_near_degree", "clutter_middle_degree", "clutter_far_degree",
    ]]

    normalized = pd.DataFrame({"Frame": frames})
    for level in range(1, 11):
        for tag_idx, tag in enumerate(TAGS):
            normalized[f"{tag}.level{level}"] = levels[:, level - 1, tag_idx]
    return normalized, individual, comprehensive


def clean_trajectory(site: str, segment: int) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    result_root = SOURCE / site / "result"
    raw = pd.read_csv(result_root / f"{segment}.csv")
    trajectory = pd.read_csv(result_root / f"result{segment}" / "main_trajectory_points.csv").iloc[:150].copy()
    if len(trajectory) != 150:
        raise ValueError(f"{site}-{segment}: main trajectory does not contain 150 rows")
    removed = int((trajectory["velocity"] > 2000).sum())
    trajectory.loc[trajectory["velocity"] > 2000, "velocity"] = np.nan
    # Preserve 150 fixed positions after excluding implausible values.
    trajectory["velocity"] = trajectory["velocity"].interpolate(limit_direction="both")
    trajectory_raw = trajectory.copy()
    trajectory["velocity"] = moving_average(trajectory["velocity"].to_numpy(), 5)
    trajectory["deviation"] = moving_average(trajectory["deviation"].to_numpy(), 5)
    return trajectory_raw, trajectory, len(raw), int(raw["ID"].nunique()), removed


def smooth_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col != "Frame":
            out[col] = moving_average(out[col].to_numpy(), 5)
    return out


def load_all_data() -> tuple[list[Dataset], pd.DataFrame]:
    datasets = []
    inventory = []
    for site, info in SITES.items():
        for segment in range(1, info["segments"] + 1):
            scene_path = SOURCE / site / "scene" / f"{segment}trim.csv"
            frames, levels, scene_meta = read_scene_values(scene_path)
            normalized, individual_raw, comprehensive_raw = aggregate_scene(frames, levels)
            trajectory_raw, trajectory, raw_points, trajectory_ids, removed = clean_trajectory(site, segment)
            individual = smooth_features(individual_raw)
            comprehensive = smooth_features(comprehensive_raw)

            segment_out = DATA_OUT / site / f"segment{segment}"
            segment_out.mkdir(parents=True, exist_ok=True)
            normalized.to_csv(segment_out / "scene_header_repaired_150.csv", index=False, encoding="utf-8-sig")
            individual_raw.to_csv(segment_out / "individual_features_raw_150.csv", index=False, encoding="utf-8-sig")
            comprehensive_raw.to_csv(segment_out / "comprehensive_features_raw_150.csv", index=False, encoding="utf-8-sig")
            individual.to_csv(segment_out / "individual_features_smoothed_150.csv", index=False, encoding="utf-8-sig")
            comprehensive.to_csv(segment_out / "comprehensive_features_smoothed_150.csv", index=False, encoding="utf-8-sig")
            trajectory_raw.to_csv(segment_out / "trajectory_cleaned_150.csv", index=False, encoding="utf-8-sig")
            trajectory.to_csv(segment_out / "trajectory_smoothed_150.csv", index=False, encoding="utf-8-sig")

            datasets.append(Dataset(
                site=site, segment=segment, frame=frames,
                individual_raw=individual_raw, comprehensive_raw=comprehensive_raw,
                individual=individual, comprehensive=comprehensive,
                trajectory_raw=trajectory_raw, trajectory=trajectory,
                raw_points=raw_points, trajectory_ids=trajectory_ids,
                velocity_removed=removed,
            ))
            inventory.append({
                "site": site, "station": info["label"], "segment": segment,
                "pedestrian_raw_points": raw_points, "trajectory_ids": trajectory_ids,
                "scene_source_rows": 160, "analysis_rows": 150,
                "source_header_fields": scene_meta["source_header_fields"],
                "discarded_empty_header_fields": scene_meta["discarded_header_fields"],
                "velocity_values_removed_above_2m_s": removed,
            })
    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(RESULTS_OUT / "data_inventory.csv", index=False, encoding="utf-8-sig")
    return datasets, inventory_df


def eligible_columns(X: pd.DataFrame) -> list[str]:
    cols = []
    for col in X.columns:
        values = X[col].to_numpy(float)
        if np.nanvar(values) <= 1e-10:
            continue
        if np.mean(np.abs(values) > 1e-12) < 0.05:
            continue
        cols.append(col)
    return cols


def select_mrmr(X: pd.DataFrame, y: np.ndarray, n_features: int = 12) -> list[str]:
    candidates = eligible_columns(X)
    if len(candidates) <= n_features:
        return candidates
    Xi = X[candidates].to_numpy(float)
    relevance = mutual_info_regression(Xi, y, random_state=42)
    if np.allclose(relevance.max(), 0):
        relevance_norm = np.zeros_like(relevance)
    else:
        relevance_norm = relevance / relevance.max()
    corr = np.abs(np.corrcoef(Xi, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    selected_idx = [int(np.argmax(relevance_norm))]
    while len(selected_idx) < min(n_features, len(candidates)):
        remaining = [i for i in range(len(candidates)) if i not in selected_idx]
        scores = []
        for i in remaining:
            redundancy = float(np.mean(corr[i, selected_idx]))
            scores.append(relevance_norm[i] - redundancy)
        selected_idx.append(remaining[int(np.argmax(scores))])
    return [candidates[i] for i in selected_idx]


def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(**RF_PARAMS)


def cv_splits(n: int, groups: np.ndarray | None):
    # Reproduce the original within-dataset five-fold evaluation logic while
    # fixing the split for exact reruns. Spatially adjacent points may occur in
    # different folds; this estimates interpolation, not transfer to new segments.
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    return list(splitter.split(np.arange(n)))


def fit_and_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_type: str,
    groups: np.ndarray | None,
) -> tuple[dict, pd.DataFrame, list[str], RandomForestRegressor, np.ndarray]:
    predictions = np.full(len(y), np.nan)
    fold_rows = []
    splits = cv_splits(len(y), groups)
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        if feature_type == "individual":
            selected = select_mrmr(X.iloc[train_idx], y[train_idx], 12)
        else:
            selected = list(X.columns)
        model = make_model()
        model.fit(X.iloc[train_idx][selected], y[train_idx])
        pred = model.predict(X.iloc[test_idx][selected])
        predictions[test_idx] = pred
        fold_rows.append({
            "fold": fold, "train_n": len(train_idx), "test_n": len(test_idx),
            "r2": r2_score(y[test_idx], pred),
            "mae": mean_absolute_error(y[test_idx], pred),
            "rmse": math.sqrt(mean_squared_error(y[test_idx], pred)),
            "selected_features": "|".join(selected),
        })
    valid = np.isfinite(predictions)
    performance = {
        "n_total": len(y), "n_oof": int(valid.sum()),
        "cv_method": "random_5fold_seed42",
        "cv_r2": r2_score(y[valid], predictions[valid]),
        "cv_mae": mean_absolute_error(y[valid], predictions[valid]),
        "cv_rmse": math.sqrt(mean_squared_error(y[valid], predictions[valid])),
        "fold_r2_mean": float(np.mean([r["r2"] for r in fold_rows])),
        "fold_r2_sd": float(np.std([r["r2"] for r in fold_rows], ddof=1)),
    }
    final_selected = select_mrmr(X, y, 12) if feature_type == "individual" else list(X.columns)
    final_model = make_model()
    final_model.fit(X[final_selected], y)
    final_pred = final_model.predict(X[final_selected])
    performance["training_r2"] = r2_score(y, final_pred)
    explainer = shap.TreeExplainer(final_model)
    shap_values = np.asarray(explainer.shap_values(X[final_selected]))
    importance = pd.DataFrame({
        "feature": final_selected,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return performance, pd.DataFrame(fold_rows), final_selected, final_model, shap_values


def scope_data(datasets: list[Dataset], site: str, segment: int | None, feature_type: str, target: str):
    chosen = [d for d in datasets if d.site == site and (segment is None or d.segment == segment)]
    feature_frames, ys, groups = [], [], []
    for d in chosen:
        feature_frames.append(d.individual.drop(columns="Frame") if feature_type == "individual" else d.comprehensive.drop(columns="Frame"))
        ys.append(d.trajectory[target].to_numpy(float))
        groups.extend([d.segment] * 150)
    X = pd.concat(feature_frames, ignore_index=True).fillna(0)
    y = np.concatenate(ys)
    return X, y, (None if segment is not None else np.asarray(groups))


def run_models(datasets: list[Dataset]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_rows = []
    selection_rows = []
    importance_rows = []
    for site, info in SITES.items():
        scopes = [(f"segment{i}", i) for i in range(1, info["segments"] + 1)] + [("total", None)]
        for scope, segment in scopes:
            for feature_type in ("individual", "comprehensive"):
                for target in ("velocity", "deviation"):
                    X, y, groups = scope_data(datasets, site, segment, feature_type, target)
                    perf, folds, selected, model, shap_values = fit_and_evaluate(X, y, feature_type, groups)
                    model_dir = MODELS_OUT / site / scope
                    model_dir.mkdir(parents=True, exist_ok=True)
                    importance = pd.DataFrame({
                        "feature": selected,
                        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
                    importance["rank"] = np.arange(1, len(importance) + 1)
                    importance.to_csv(model_dir / f"{feature_type}_{target}_shap_importance.csv", index=False, encoding="utf-8-sig")
                    folds.to_csv(model_dir / f"{feature_type}_{target}_cv_folds.csv", index=False, encoding="utf-8-sig")
                    pd.DataFrame({"observed": y, "training_prediction": model.predict(X[selected])}).to_csv(
                        model_dir / f"{feature_type}_{target}_training_predictions.csv", index=False, encoding="utf-8-sig"
                    )
                    performance_rows.append({
                        "site": site, "station": info["label"], "scope": scope,
                        "segment": segment if segment is not None else "total",
                        "feature_type": feature_type, "target": target,
                        **perf,
                    })
                    for rank, feature in enumerate(selected, start=1):
                        selection_rows.append({
                            "site": site, "scope": scope, "feature_type": feature_type,
                            "target": target, "rank_in_selected_set": rank, "feature": feature,
                        })
                    for row in importance.to_dict("records"):
                        importance_rows.append({
                            "site": site, "scope": scope, "segment": segment,
                            "feature_type": feature_type, "target": target, **row,
                        })
    performance_df = pd.DataFrame(performance_rows)
    selections_df = pd.DataFrame(selection_rows)
    importance_df = pd.DataFrame(importance_rows)
    performance_df.to_csv(RESULTS_OUT / "model_performance.csv", index=False, encoding="utf-8-sig")
    selections_df.to_csv(RESULTS_OUT / "selected_features.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(RESULTS_OUT / "all_shap_importance.csv", index=False, encoding="utf-8-sig")
    return performance_df, selections_df, importance_df


def descriptive_tables(datasets: list[Dataset]) -> pd.DataFrame:
    rows = []
    for site, info in SITES.items():
        chosen = [d for d in datasets if d.site == site]
        for d in chosen:
            tr = d.trajectory_raw
            rows.append({
                "site": site, "station": info["label"], "sample_set": str(d.segment),
                "trajectory_ids": d.trajectory_ids, "aggregation_points": len(tr),
                "velocity_mean_m_s": tr.velocity.mean() / 1000,
                "velocity_sd_m_s": tr.velocity.std(ddof=1) / 1000,
                "deviation_mean_m": tr.deviation.mean() / 1000,
                "deviation_sd_m": tr.deviation.std(ddof=1) / 1000,
            })
        all_tr = pd.concat([d.trajectory_raw for d in chosen], ignore_index=True)
        rows.append({
            "site": site, "station": info["label"], "sample_set": "Total",
            "trajectory_ids": sum(d.trajectory_ids for d in chosen),
            "aggregation_points": len(all_tr),
            "velocity_mean_m_s": all_tr.velocity.mean() / 1000,
            "velocity_sd_m_s": all_tr.velocity.std(ddof=1) / 1000,
            "deviation_mean_m": all_tr.deviation.mean() / 1000,
            "deviation_sd_m": all_tr.deviation.std(ddof=1) / 1000,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_OUT / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    return df


def depth_of(feature: str) -> str | None:
    if ".near" in feature or "_near_" in feature or feature.startswith("near_"):
        return "near"
    if ".middle" in feature or "_middle_" in feature or feature.startswith("middle_"):
        return "middle"
    if ".far" in feature or "_far_" in feature or feature.startswith("far_"):
        return "far"
    return None


def aggregate_rankings(importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = importance[importance["scope"].str.startswith("segment")].copy()
    local["model_id"] = (
        local["site"].astype(str) + "|" + local["scope"].astype(str) + "|"
        + local["feature_type"].astype(str) + "|" + local["target"].astype(str)
    )
    model_totals = local.groupby("model_id")["mean_abs_shap"].transform("sum")
    if (model_totals <= 0).any():
        raise ValueError("A segment model has non-positive total mean absolute SHAP importance.")
    # Normalize within each fitted model so velocity and deviation models are
    # comparable despite different target units. A feature absent from a model
    # is represented implicitly by zero when the normalized shares are averaged
    # over every eligible model rather than only over appearances.
    local["normalized_shap_share"] = local["mean_abs_shap"] / model_totals
    local["depth"] = local["feature"].map(depth_of)

    table5_denominators = (
        local[["target", "feature_type", "model_id"]].drop_duplicates()
        .groupby(["target", "feature_type"]).size().rename("eligible_models").reset_index()
    )
    table5 = (local.groupby(["target", "feature_type", "depth", "feature"], dropna=False)
              .agg(shap_share_sum=("normalized_shap_share", "sum"),
                   appearances=("model_id", "nunique"))
              .reset_index()
              .merge(table5_denominators, on=["target", "feature_type"], how="left"))
    table5["score"] = table5["shap_share_sum"] / table5["eligible_models"]
    table5["appearance_rate"] = table5["appearances"] / table5["eligible_models"]
    table5["conditional_shap_share"] = table5["shap_share_sum"] / table5["appearances"]
    table5 = table5.drop(columns="shap_share_sum")
    table5 = table5.sort_values(["target", "feature_type", "depth", "score"], ascending=[True, True, True, False])
    table5.to_csv(RESULTS_OUT / "table5_depth_rankings.csv", index=False, encoding="utf-8-sig")

    def position(site: str, segment: int) -> str:
        if site == "wujiaochang":
            return "begin" if segment <= 2 else ("middle" if segment == 3 else "end")
        # East Nanjing Road: segments 1-4 belong to Line 10; 5-7 to Line 2.
        if segment in (1, 5):
            return "begin"
        if segment in (2, 3, 6):
            return "middle"
        return "end"

    local["segment_int"] = local["scope"].str.replace("segment", "", regex=False).astype(int)
    local["position"] = [position(s, i) for s, i in zip(local.site, local.segment_int)]
    table6_denominators = (
        local[["position", "feature_type", "model_id"]].drop_duplicates()
        .groupby(["position", "feature_type"]).size().rename("eligible_models").reset_index()
    )
    table6 = (local.groupby(["position", "feature_type", "depth", "feature"], dropna=False)
              .agg(shap_share_sum=("normalized_shap_share", "sum"),
                   appearances=("model_id", "nunique"))
              .reset_index()
              .merge(table6_denominators, on=["position", "feature_type"], how="left"))
    table6["score"] = table6["shap_share_sum"] / table6["eligible_models"]
    table6["appearance_rate"] = table6["appearances"] / table6["eligible_models"]
    table6["conditional_shap_share"] = table6["shap_share_sum"] / table6["appearances"]
    table6 = table6.drop(columns="shap_share_sum")
    table6 = table6.sort_values(["position", "feature_type", "depth", "score"], ascending=[True, True, True, False])
    table6.to_csv(RESULTS_OUT / "table6_position_rankings.csv", index=False, encoding="utf-8-sig")
    return table5, table6


def readable_feature(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return (feature.replace(".near", " (proximal)")
            .replace(".middle", " (medial)")
            .replace(".far", " (distal)")
            .replace("_", " "))


def plot_global_top5(importance: pd.DataFrame, feature_type: str, target: str, figure_number: int):
    subset = importance[(importance.scope == "total") & (importance.feature_type == feature_type) & (importance.target == target)]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    color = "#3B6EA8" if target == "velocity" else "#E64B35"
    edge_color = "#263238" if target == "velocity" else "#8B2F28"
    for ax, site in zip(axes, ("wujiaochang", "nanjingdonglu")):
        station = SITES[site]["label"]
        data = subset[subset.site == site].nsmallest(5, "rank").sort_values("mean_abs_shap")
        labels = [readable_feature(x) for x in data.feature]
        ax.barh(labels, data.mean_abs_shap, color=color, edgecolor=edge_color, linewidth=0.6)
        ax.set_title(station, fontsize=11, weight="bold")
        unit = "mm/s" if target == "velocity" else "mm"
        ax.set_xlabel(f"Mean |SHAP value| ({unit})", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    title_type = "Individual visual variables" if feature_type == "individual" else "Comprehensive visual variables"
    fig.suptitle(f"{title_type}: top five predictors of {target}", fontsize=12, weight="bold")
    stem = FIG_OUT / f"figure{figure_number}_{feature_type}_{target}_top5"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(importance: pd.DataFrame):
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    plot_global_top5(importance, "individual", "velocity", 10)
    plot_global_top5(importance, "comprehensive", "velocity", 11)
    plot_global_top5(importance, "individual", "deviation", 12)
    plot_global_top5(importance, "comprehensive", "deviation", 13)


def build_summary(
    inventory: pd.DataFrame,
    performance: pd.DataFrame,
    descriptive: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
) -> dict:
    totals = inventory.groupby("site").agg(
        raw_points=("pedestrian_raw_points", "sum"),
        trajectory_ids=("trajectory_ids", "sum"),
        analysis_rows=("analysis_rows", "sum"),
        removed_velocity=("velocity_values_removed_above_2m_s", "sum"),
    ).to_dict("index")

    perf_total = performance[performance.scope == "total"].copy()
    performance_records = perf_total.to_dict("records")

    def top_record(frame: pd.DataFrame, filters: dict):
        x = frame.copy()
        for key, val in filters.items():
            x = x[x[key] == val]
        if x.empty:
            return None
        row = x.sort_values("score", ascending=False).iloc[0]
        return {"feature": row.feature, "score": float(row.score), "appearances": int(row.appearances)}

    depth_top = {}
    for target in ("velocity", "deviation"):
        depth_top[target] = {}
        for ft in ("individual", "comprehensive"):
            depth_top[target][ft] = {}
            for depth in ("near", "middle", "far"):
                depth_top[target][ft][depth] = top_record(table5, {"target": target, "feature_type": ft, "depth": depth})

    position_top = {}
    for pos in ("begin", "middle", "end"):
        position_top[pos] = {}
        for ft in ("individual", "comprehensive"):
            position_top[pos][ft] = {}
            for depth in ("near", "middle", "far"):
                position_top[pos][ft][depth] = top_record(table6, {"position": pos, "feature_type": ft, "depth": depth})

    summary = {
        "totals": totals,
        "descriptive": descriptive.to_dict("records"),
        "total_model_performance": performance_records,
        "depth_top": depth_top,
        "position_top": position_top,
        "rf_params": RF_PARAMS,
        "figures": {str(i): str(next(FIG_OUT.glob(f"figure{i}_*.png"))) for i in range(10, 14)},
    }
    with (RESULTS_OUT / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    return summary


def main():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    datasets, inventory = load_all_data()
    descriptive = descriptive_tables(datasets)
    performance, selections, importance = run_models(datasets)
    table5, table6 = aggregate_rankings(importance)
    create_figures(importance)
    summary = build_summary(inventory, performance, descriptive, table5, table6)
    print(json.dumps({
        "datasets": len(datasets),
        "inventory_rows": len(inventory),
        "models": len(performance),
        "figures": summary["figures"],
        "total_performance": summary["total_model_performance"],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
