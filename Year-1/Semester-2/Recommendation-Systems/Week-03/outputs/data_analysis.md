# Data Analysis Summary

## Dataset shape
- `ratings_train.csv`: 862,335 rows, 4 columns
- `ratings_to_predict.csv`: 287,445 rows, 3 columns
- `users.csv`: 278,858 rows, 3 columns
- `books.csv`: 271,376 rows, 5 columns

## What stands out
- The interaction matrix is extremely sparse (`~99.9966%` missing user-book cells).
- Ratings are very skewed toward zero (62.28% of train ratings are `0`).
- Cold-start exists in test:
  - unseen users: 5.89% of test rows
  - unseen books: 20.23% of test rows
- User activity is long-tailed:
  - median ratings/user: 1
  - 99th percentile ratings/user: 151
- Book popularity is long-tailed:
  - median ratings/book: 1
  - 99th percentile ratings/book: 31

## Cleaning / normalization decisions
- `Age`: values outside `[5, 100]` treated as invalid (40.17%); imputed with median and flagged with `Age_Missing`.
- `Year`: values outside `[1800, 2026]` treated as invalid (1.71%); imputed with median and flagged with `Year_Missing`.
- Categorical missing/blank values (`Country`, `Author`, `Publisher`) filled with `Unknown`.
- No z-score normalization was needed for tree-based final model.
- Instead of one-hot encoding huge categorical spaces, smoothed target statistics were used (`author_mean`, `publisher_mean`, `country_mean`, etc.).

## Modeling approach
- Collaborative filtering baseline: regularized user/book bias model.
- Hybrid model: gradient boosting over CF prediction + user/book stats + metadata stats.
- Validation RMSE:
  - global mean baseline: 3.85673
  - bias-only CF: 3.41020
  - hybrid model: 3.81824
- Selected model for submission: `bias` (lowest validation RMSE).
