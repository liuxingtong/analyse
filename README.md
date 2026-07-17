# Depth-dependent visual perception and pedestrian movement in metro walkways

This repository contains the curated data, reproducible Python workflow, model outputs, and figures used to analyze nonlinear associations between depth-partitioned visual elements and pedestrian movement in metro walkways.

The study includes 12 matched walkway segments from two Shanghai metro stations:

- Wujiaochang: 5 segments, 750 analysis points, 191 segment-specific trajectory IDs.
- East Nanjing Road: 7 segments, 1,050 analysis points, 275 segment-specific trajectory IDs.
- Total: 1,800 matched analysis points, 466 segment-specific trajectory IDs, and 49,138 raw pedestrian trajectory points before spatial aggregation.

## Repository structure

```text
data/
  cleaned/                  # Seven cleaned 150-row files for every segment
  metadata/                 # Inventory, sample sizes, units, and descriptive statistics
docs/
  METHODS.md                # Complete processing and modeling specification
  DATA_DICTIONARY.md        # Columns and file-level definitions
  DATA_LOCATIONS.md         # Original local sources and repository destinations
figures/                    # Figures 10–13 in PNG, SVG, and PDF
results/
  summary/                  # Performance, selected variables, rankings, and validation
  models/                   # Fold metrics, training predictions, and SHAP importance
scripts/
  build_cleaned_and_analyze.py  # Rebuild from external raw source data
  run_models_from_cleaned.py    # Reproduce models and figures from included data
  compare_cv_methods.py         # Compare contiguous and random five-fold schemes
  validate_repository.py        # Check repository completeness and data integrity
```

Legacy scripts, raw videos, raw tracking exports, manuscripts, Office files, previews, caches, and local credentials are excluded by the allow-list `.gitignore`.

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_repository.py
python scripts/run_models_from_cleaned.py
```

The second command validates 84 cleaned segment files, 56 model records, 168 per-model result files, and 12 exported figure files. The third command reruns mRMR selection, Random Forest models, SHAP importance, weighted rankings, and Figures 10–13 from the included cleaned data.

## Rebuilding from raw local data

Raw tracking and scene files are intentionally not versioned. To rebuild cleaned data from an external source directory containing `nanjingdonglu/` and `wujiaochang/`, set `DEPTH_SOURCE_ROOT` and run:

```powershell
$env:DEPTH_SOURCE_ROOT = 'D:\path\to\raw\analyse'
python scripts/build_cleaned_and_analyze.py
```

Without the environment variable, the script looks for the raw station directories at the repository root. See [docs/DATA_LOCATIONS.md](docs/DATA_LOCATIONS.md) for the original workstation mapping.

## Analysis specification

- Each segment retains the first 150 matched points at 50 mm spacing.
- Velocity values above 2 m/s are removed and interpolated according to the manuscript cleaning rule.
- No additional predictor standardization is applied.
- Scene and trajectory series use a five-point moving average.
- Individual variables pass variance/sparsity screening and fold-specific mRMR selection to 12 predictors.
- All 12 theory-derived comprehensive variables enter the model.
- Random Forest parameters are fixed at 100 trees, maximum depth 10, minimum split size 10, minimum leaf size 5, `max_features="sqrt"`, and seed 42.
- Reported performance uses shuffled random five-fold cross-validation with seed 42.
- SHAP values describe attribution within the fitted Random Forest and are not significance tests or causal effects.

The station-level cross-validated R² values range from 0.346 to 0.868. Random folds estimate within-dataset interpolation; spatial smoothing means adjacent points can enter different folds, so performance may be optimistic for transfer to unseen segments.

Full details are in [docs/METHODS.md](docs/METHODS.md).

## Data units

- Published descriptive velocity: m/s.
- Published descriptive deviation: m.
- Model targets and SHAP magnitudes: mm/s for velocity and mm for deviation.
- Scene variables: semantic pixel proportions or ratios, as defined in the data dictionary.

## Reproducibility status

The included repository validation report is at `results/summary/repository_validation.json`. Figures are provided in editable vector formats (`.svg`, `.pdf`) and 600 dpi raster format (`.png`).
