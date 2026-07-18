# Processing and analysis methods

## 1. Source matching

Pedestrian and scene files are matched by station and segment number. East Nanjing Road uses segments 1–7; Wujiaochang uses segments 1–5. A scene file and its pedestrian result file must share the same number.

## 2. Pedestrian trajectory processing

Retained pedestrian trajectory observations are ordered along the principal walkway axis and grouped using a 50 mm coordinate tolerance. The mean coordinates of each group define one representative spatial analysis location. The `point_count` field records the number of raw trajectory coordinate observations grouped at that location. Velocity is calculated separately as the mean point velocity within an overlapping 1 m circular neighborhood, so `point_count` should not be interpreted as the velocity-neighborhood sample size. Deviation is the shortest distance from the spatial location to the first principal-component reference line.

The curated inventory reports 29,981 raw trajectory points and 275 segment-specific IDs at East Nanjing Road, plus 19,157 points and 191 IDs at Wujiaochang. IDs are unique within source files and are not assumed to identify the same person across segments.

Velocity values above 2 m/s are removed under the original cleaning rule and interpolated along ordered points. This removes 11 values at East Nanjing Road and 5 at Wujiaochang. The filter does not identify walking motivation and should not be interpreted as proving leisure walking.

## 3. Scene header repair

Every scene contains 10 depth levels and 53 semantic classes. Wujiaochang scene files 1–4 contained duplicated header fields and empty trailing columns. The repair procedure:

1. reads the first 541 meaningful source fields;
2. removes the marker field at the start of each 54-field depth block;
3. assigns a standardized `semantic_class.levelN` header to the remaining values;
4. preserves all numeric scene values exactly.

## 4. Spatial alignment and smoothing

The first 150 ordered spatial analysis locations are retained for every pedestrian and scene segment. These locations are generated using the 50 mm principal-axis grouping tolerance and are matched to the first 150 scene rows by segment and order. Both predictor and target series receive the same centered five-point moving average used in the manuscript workflow. No additional predictor standardization is performed.

Standardization is unnecessary for the present Random Forest workflow because tree splits depend on within-feature ordering and are invariant to monotonic rescaling of individual predictors. Pearson redundancy is also invariant to linear rescaling, and the mRMR mutual-information term measures dependence rather than coefficient magnitude. Retaining the original scale preserves interpretable target and SHAP units. This rationale would not automatically apply to scale-sensitive models such as KNN, SVM, or regularized linear regression.

## 5. Depth aggregation

The 10 scene levels are aggregated into:

- proximal: levels 1, representing 0–2 m;
- medial: levels 2–5, representing 2–10 m;
- distal: levels 6–10, representing 10–20 m.

These boundaries are context-specific and were not subjected to sensitivity analysis in the present workflow.

## 6. Predictor construction

Individual predictors are semantic class proportions within each of the three depth zones. Before modeling, variables with variance at or below `1e-10` or a non-zero rate below 5% are removed.

Twelve individual predictors are selected by minimum-redundancy maximum-relevance (mRMR). Relevance is mutual information with the target. Redundancy is the mean absolute Pearson correlation with already selected variables. Selection is repeated inside each cross-validation training fold.

The 12 comprehensive predictors are:

- proximal, medial, and distal area ratios;
- obstacle density in each zone, using fence, railing, column, bar, and bannister;
- visibility ratio in each zone, using glass and windowpane;
- clutter degree in each zone, defined as the non-structural share after excluding wall, floor, ceiling, stairs, and stairway.

All comprehensive predictors enter each comprehensive-variable model.

## 7. Random Forest models

Separate models are fitted for velocity and deviation using individual and comprehensive predictor sets. Models are estimated for each segment and for the combined data at each station, producing 56 model records.

Fixed parameters:

```text
n_estimators       = 100
max_depth          = 10
min_samples_split  = 10
min_samples_leaf   = 5
max_features       = sqrt
random_state       = 42
```

## 8. Cross-validation

The reported analysis uses shuffled five-fold KFold cross-validation with seed 42. For individual-variable models, mRMR is fitted only on the training rows in each fold. Pooled out-of-fold R², MAE, and RMSE are reported. Training R² is retained as a diagnostic.

Random folds estimate interpolation within the observed datasets. Because moving averages introduce spatial dependence, adjacent points can be assigned to different folds and may produce optimistic estimates for unseen walkway segments. `compare_cv_methods.py` records the contrast with contiguous default KFold evaluation.

## 9. SHAP and stability-adjusted synthesis

TreeSHAP is calculated from the final Random Forest fitted to the complete analysis set. Importance is the mean absolute SHAP value. Figures 10–13 show the five highest station-level attributions for each predictor-set/outcome combination.

For depth and walkway-position synthesis, each feature's mean absolute SHAP value is divided by the sum of mean absolute SHAP values for all selected predictors in the same fitted model. This produces a model-normalized SHAP contribution share and permits synthesis across velocity and deviation models despite their different target units. A feature absent after screening or mRMR selection receives a share of zero. The stability-adjusted SHAP score is the mean normalized share across all eligible models: 12 segment models for each outcome in the depth-zone synthesis and eight segment-outcome models for each walkway position. The score therefore reflects both attribution magnitude when present and recurrence across models. Appearance counts and rates are retained separately.

SHAP is an explanation of the fitted predictive model. It is not a p-value, confidence interval, conventional effect size, or causal estimate.

## 10. Known limitations

- Main-trajectory aggregation suppresses between-person variation.
- The speed threshold does not establish leisure-walking motivation.
- Crowd density, time context, and visual-element interactions are not modeled as covariates.
- Depth boundaries are not sensitivity-tested.
- Random-fold performance does not establish transfer to unseen segments or independent datasets.
- Observational Random Forest and SHAP results do not establish causality.
