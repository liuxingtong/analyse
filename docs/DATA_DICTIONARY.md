# Data dictionary

## Segment directory layout

Each directory under `data/cleaned/{site}/segmentN/` contains 150 ordered spatial aggregation locations. After coordinate registration, a 7,500 mm corridor segment is sampled every 50 mm, and the spatial-scene video frame corresponding to each location supplies the matched visual variables. The number 150 is therefore a fixed spatial sampling design, not the number of independent pedestrians or a truncation of scene-variable rows.

| File | Grain | Description |
|---|---:|---|
| `scene_header_repaired_150.csv` | 150 × 531 | Frame index plus 530 values: 53 semantic classes at 10 depth levels. |
| `individual_features_raw_150.csv` | 150 × 160 | Frame index plus 53 semantic classes aggregated into proximal, medial, and distal zones. |
| `individual_features_smoothed_150.csv` | 150 × 160 | Five-point moving-average version used by the models. |
| `comprehensive_features_raw_150.csv` | 150 × 13 | Frame index plus 12 theory-derived comprehensive predictors. |
| `comprehensive_features_smoothed_150.csv` | 150 × 13 | Five-point moving-average version used by the models. |
| `trajectory_cleaned_150.csv` | 150 × 5 | Cleaned `x`, `y`, `velocity`, `point_count`, and `deviation`. |
| `trajectory_smoothed_150.csv` | 150 × 5 | Five-point moving-average trajectory targets used by the models. |

## Sites

| Repository key | Published label | Segments | Analysis rows |
|---|---|---:|---:|
| `nanjingdonglu` | East Nanjing Road | 7 | 1,050 |
| `wujiaochang` | Wujiaochang | 5 | 750 |

## Individual feature naming

Individual columns use `{semantic_class}.{zone}`. Zone suffixes are:

- `.near`: proximal, 0–2 m;
- `.middle`: medial, 2–10 m;
- `.far`: distal, 10–20 m.

Values are aggregated semantic pixel proportions. They are not standardized before modeling.

## Comprehensive variables

| Column | Definition |
|---|---|
| `near_ratio` | Recognized proximal area divided by recognized area across all zones. |
| `middle_ratio` | Recognized medial area divided by recognized area across all zones. |
| `far_ratio` | Recognized distal area divided by recognized area across all zones. |
| `obstacle_{zone}_density` | Fence, railing, column, bar, and bannister area divided by recognized zone area. |
| `visibility_{zone}_ratio` | Glass and windowpane area divided by recognized zone area. |
| `clutter_{zone}_degree` | Non-structural area divided by recognized zone area. Structural classes are wall, floor, ceiling, stairs, and stairway. |

## Trajectory columns and units

| Column | Unit | Definition |
|---|---|---|
| `x`, `y` | mm | Representative trajectory coordinates. |
| `velocity` | mm/s | Aggregated pedestrian velocity; divide by 1,000 for m/s. |
| `point_count` | count | Raw trajectory coordinate observations grouped with the spatial location under the 50 mm principal-axis tolerance. This is not the count in the overlapping 1,000 mm neighborhood used to calculate velocity. |
| `deviation` | mm | Distance from the PCA reference path; divide by 1,000 for m. |

## Metadata files

- `data_inventory.csv`: station, segment, raw point count, trajectory-ID count, source rows, analysis rows, header repair, and speed-removal counts.
- `descriptive_statistics.csv`: sample sizes and mean/SD values in m/s and m.

## Result files

- `model_performance.csv`: cross-validation and training metrics for 56 models.
- `selected_features.csv`: final mRMR-selected individual features and the comprehensive predictors.
- `all_shap_importance.csv`: mean absolute SHAP importance and ranks.
- `table5_depth_rankings.csv`: weighted rankings by outcome, variable type, and depth.
- `table6_position_rankings.csv`: weighted rankings by walkway position, variable type, and depth.
- `cv_method_comparison.csv`: random versus contiguous five-fold performance comparison.
- `trajectory_aggregation_support.csv`: all 1,800 retained spatial locations with station, segment, location index, coordinates, velocity, deviation, and `point_count`.
- `trajectory_aggregation_summary.csv`: segment-level mean, quartiles, median, minimum, and maximum of `point_count`.
- `analysis_summary.json`: machine-readable headline results and parameters.
- `repository_validation.json`: repository integrity check.

Each directory in `results/models/{site}/{scope}/` contains fold metrics, full-data training predictions, and SHAP importance for the four predictor-set/outcome combinations.
