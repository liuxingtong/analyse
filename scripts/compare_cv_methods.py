from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import build_cleaned_and_analyze as ra

OUT = Path(__file__).resolve().parents[1] / "results" / "summary" / "cv_method_comparison.csv"


def evaluate(X, y, feature_type, splitter):
    pred = np.full(len(y), np.nan)
    fold_r2 = []
    for train_idx, test_idx in splitter.split(X):
        selected = ra.select_mrmr(X.iloc[train_idx], y[train_idx], 12) if feature_type == "individual" else list(X.columns)
        model = ra.make_model()
        model.fit(X.iloc[train_idx][selected], y[train_idx])
        fold_pred = model.predict(X.iloc[test_idx][selected])
        pred[test_idx] = fold_pred
        fold_r2.append(r2_score(y[test_idx], fold_pred))
    return {
        "pooled_oof_r2": r2_score(y, pred),
        "mean_fold_r2": np.mean(fold_r2),
        "sd_fold_r2": np.std(fold_r2, ddof=1),
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
    }


def main():
    datasets, _ = ra.load_all_data()
    rows = []
    methods = {
        "original_default_kfold": KFold(n_splits=5, shuffle=False),
        "random_kfold_seed42": KFold(n_splits=5, shuffle=True, random_state=42),
    }
    for site, info in ra.SITES.items():
        for feature_type in ("individual", "comprehensive"):
            for target in ("velocity", "deviation"):
                X, y, _ = ra.scope_data(datasets, site, None, feature_type, target)
                for name, splitter in methods.items():
                    metrics = evaluate(X, y, feature_type, splitter)
                    rows.append({
                        "site": site,
                        "station": info["label"],
                        "feature_type": feature_type,
                        "target": target,
                        "method": name,
                        "n": len(y),
                        **metrics,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
