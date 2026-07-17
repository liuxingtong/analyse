# Data and output locations

## Repository-relative locations

- Cleaned 150-point data: `data/cleaned/{site}/segmentN/`
- Sample inventory and descriptive statistics: `data/metadata/`
- Reproducible scripts: `scripts/`
- Summary model outputs: `results/summary/`
- Per-model outputs: `results/models/{site}/{segmentN|total}/`
- Figures 10–13: `figures/`
- Method and data documentation: `docs/`

## Original workstation sources

The raw inputs are intentionally ignored and are not required to rerun models from the included cleaned data.

- East Nanjing Road pedestrian files: `F:\Aworks\DepthPerception\1metro\analyse\nanjingdonglu\result\1.csv` through `7.csv`
- East Nanjing Road scene files: `F:\Aworks\DepthPerception\1metro\analyse\nanjingdonglu\scene\1trim.csv` through `7trim.csv`
- Wujiaochang pedestrian files: `F:\Aworks\DepthPerception\1metro\analyse\wujiaochang\result\1.csv` through `5.csv`
- Wujiaochang scene files: `F:\Aworks\DepthPerception\1metro\analyse\wujiaochang\scene\1trim.csv` through `5trim.csv`

Segment numbers match one-to-one within each station.

## Source analysis workspace

- Original revision workspace: `F:\Aworks\DepthPerception\1metro\revision_analysis`
- Original cleaned-data directory: `F:\Aworks\DepthPerception\1metro\revision_analysis\cleaned_data`
- Original results directory: `F:\Aworks\DepthPerception\1metro\revision_analysis\results`
- Original figure directory: `F:\Aworks\DepthPerception\1metro\revision_analysis\figures`

## Files excluded from Git

The repository does not version raw videos, full tracking dumps, Office manuscripts, slide decks, 3D model files, local previews, caches, virtual environments, or secrets. These files remain in the original workstation directories and are protected by the allow-list `.gitignore`.
