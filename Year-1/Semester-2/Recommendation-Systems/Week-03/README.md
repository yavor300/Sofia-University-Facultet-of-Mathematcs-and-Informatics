# FMI SU Recommender System HW 3 (2026)

This repository contains a complete local solution pipeline for the Kaggle competition:
`fmi-su-recommender-system-hw-3-2026`.

The main script is:
- `kaggle_solution.py`

## 1. Solution Architecture

The pipeline is built as a **hybrid recommender system** with model selection:

1. Load raw data
- `data/ratings_train.csv`
- `data/ratings_to_predict.csv`
- `data/users.csv`
- `data/books.csv`

2. Clean and enrich metadata
- Users:
  - Clean `Country` (`Unknown` fallback).
  - Validate `Age` in `[5, 100]`; invalid values are treated as missing.
  - Impute missing age with median.
  - Add `Age_Missing` flag and `Age_Bin`.
- Books:
  - Clean `Title`, `Author`, `Publisher` (`Unknown` fallback).
  - Validate `Year` in `[1800, 2026]`; invalid values are treated as missing.
  - Impute missing year with median.
  - Add `Year_Missing` flag and `Year_Bin`.

3. Train collaborative filtering baseline
- Regularized user bias + book bias model:
  - `prediction = global_mean + user_bias + book_bias`
- This handles sparse interaction data effectively.

4. Train hybrid residual model
- A `HistGradientBoostingRegressor` learns the **residual error** over the bias baseline.
- Features include:
  - CF baseline prediction.
  - User/book historical stats (`mean`, `count`).
  - Smoothed target stats from metadata groups (`author`, `publisher`, `country`, age bin, year bin).
  - Cold-start flags and interaction features.

5. Validate and select best model
- Compare:
  - Global mean baseline
  - Bias-only CF
  - Hybrid model
- Use validation RMSE and pick the best model automatically for final test prediction.

6. Generate Kaggle submission
- Predict on `ratings_to_predict.csv`.
- Round ratings to integers and clip to `[0, 10]`.
- Save `outputs/submission.csv` with format:
  - `Id,Rating`

## 2. Approaches ("Grasps") Used

The solution combines several approaches:

1. Collaborative filtering (CF)
- Captures user preferences and book popularity from interactions only.

2. Content/metadata-aware signals
- Uses user (`Country`, `Age`) and book (`Author`, `Publisher`, `Year`) metadata via target statistics.

3. Hybrid modeling
- Blends CF and metadata by learning residual corrections with gradient boosting.

4. Data quality handling
- Outlier handling, imputation, and missing flags to improve robustness.

5. Validation-driven model selection
- Final submission model is chosen by measured validation RMSE, not by assumption.

## 3. Meaning of the Used Metrics

The primary metric is **RMSE (Root Mean Squared Error)**.

Formula:

`RMSE = sqrt(mean((y_true - y_pred)^2))`

Interpretation:
- RMSE measures average prediction error in the same units as the rating scale (0 to 10).
- Lower is better.
- Squaring the errors penalizes large mistakes more strongly.

In this project we report:
- `rmse_global`: error from predicting one constant rating (global average).
- `rmse_bias`: error from the collaborative filtering bias model.
- `rmse_hybrid`: error from the hybrid residual model.

Example from current run:
- `rmse_global = 3.85673`
- `rmse_bias = 3.41020`
- `rmse_hybrid = 3.81824`

This means the bias CF model currently generalizes best on validation, so it is selected for submission.

## 4. Files Produced

After running:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python kaggle_solution.py
```

You get:
- `outputs/submission.csv` (Kaggle upload file)
- `outputs/validation_metrics.json` (numeric metrics)
- `outputs/data_analysis.md` (data analysis summary)

## 5. Notes

- The dataset is extremely sparse and long-tailed, so strong regularization and robust defaults are important.
- About 20% of test rows contain books unseen in train, so metadata and fallback logic are necessary.
