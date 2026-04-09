#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

USER_COL = "User_Id"
BOOK_COL = "Book_Id"
RATING_COL = "Rating"
ID_COL = "Unnamed: 0"


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clean_users(users: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    users = users.drop_duplicates(subset=[USER_COL]).copy()
    users["Country"] = users["Country"].fillna("Unknown").astype(str).str.strip()
    users.loc[users["Country"] == "", "Country"] = "Unknown"

    valid_age = users["Age"].between(5, 100)
    users["Age_Missing"] = (~valid_age).astype(np.int8)
    users["Age_Clean"] = users["Age"].where(valid_age, np.nan)
    age_median = float(users["Age_Clean"].median())
    users["Age_Clean"] = users["Age_Clean"].fillna(age_median).astype(np.float32)
    users["Age_Bin"] = (users["Age_Clean"] // 10 * 10).astype(np.int16)

    meta = {"age_median": age_median}
    return users[[USER_COL, "Country", "Age_Clean", "Age_Missing", "Age_Bin"]], meta


def clean_books(books: pd.DataFrame, current_year: int = 2026) -> Tuple[pd.DataFrame, Dict[str, float]]:
    books = books.drop_duplicates(subset=[BOOK_COL]).copy()
    for col in ["Title", "Author", "Publisher"]:
        books[col] = books[col].fillna("Unknown").astype(str).str.strip()
        books.loc[books[col] == "", col] = "Unknown"

    year = pd.to_numeric(books["Year"], errors="coerce")
    valid_year = year.between(1800, current_year)
    books["Year_Missing"] = (~valid_year).astype(np.int8)
    books["Year_Clean"] = year.where(valid_year, np.nan)
    year_median = float(books["Year_Clean"].median())
    books["Year_Clean"] = books["Year_Clean"].fillna(year_median).astype(np.float32)
    books["Year_Bin"] = (books["Year_Clean"] // 5 * 5).astype(np.int16)

    meta = {"year_median": year_median}
    return books[[BOOK_COL, "Title", "Author", "Publisher", "Year_Clean", "Year_Missing", "Year_Bin"]], meta


def _smoothed_group_stats(
    frame: pd.DataFrame,
    key: str,
    global_mean: float,
    alpha: float = 10.0,
) -> Tuple[pd.Series, pd.Series]:
    grouped = frame.groupby(key, observed=False)[RATING_COL].agg(["sum", "count"])
    smooth_mean = ((grouped["sum"] + alpha * global_mean) / (grouped["count"] + alpha)).astype(np.float32)
    return smooth_mean, grouped["count"].astype(np.float32)


def fit_bias_terms(
    ratings: pd.DataFrame,
    global_mean: float,
    lambda_user: float = 15.0,
    lambda_book: float = 10.0,
    n_iters: int = 8,
) -> Tuple[pd.Series, pd.Series]:
    user_bias = pd.Series(0.0, index=ratings[USER_COL].unique(), dtype=np.float32)
    book_bias = pd.Series(0.0, index=ratings[BOOK_COL].unique(), dtype=np.float32)
    work = ratings[[USER_COL, BOOK_COL, RATING_COL]].copy()

    for _ in range(n_iters):
        work["book_bias"] = work[BOOK_COL].map(book_bias).fillna(0.0).astype(np.float32)
        user_residual = work[RATING_COL] - global_mean - work["book_bias"]
        user_agg = user_residual.groupby(work[USER_COL], observed=False).agg(["sum", "count"])
        user_bias = (user_agg["sum"] / (lambda_user + user_agg["count"])).astype(np.float32)

        work["user_bias"] = work[USER_COL].map(user_bias).fillna(0.0).astype(np.float32)
        book_residual = work[RATING_COL] - global_mean - work["user_bias"]
        book_agg = book_residual.groupby(work[BOOK_COL], observed=False).agg(["sum", "count"])
        book_bias = (book_agg["sum"] / (lambda_book + book_agg["count"])).astype(np.float32)

    return user_bias, book_bias


@dataclass
class FeatureState:
    global_mean: float
    user_bias: pd.Series
    book_bias: pd.Series
    user_mean: pd.Series
    user_count: pd.Series
    book_mean: pd.Series
    book_count: pd.Series
    author_mean: pd.Series
    author_count: pd.Series
    publisher_mean: pd.Series
    publisher_count: pd.Series
    country_mean: pd.Series
    country_count: pd.Series
    age_bin_mean: pd.Series
    year_bin_mean: pd.Series


class HybridBookRecommender:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.state: FeatureState | None = None
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=1.0,
            validation_fraction=0.1,
            early_stopping=True,
            random_state=random_state,
        )
        self.users_clean: pd.DataFrame | None = None
        self.books_clean: pd.DataFrame | None = None
        self.user_defaults: Dict[str, float] = {}
        self.book_defaults: Dict[str, float] = {}

    def fit(
        self,
        ratings: pd.DataFrame,
        users_clean: pd.DataFrame,
        books_clean: pd.DataFrame,
        user_defaults: Dict[str, float],
        book_defaults: Dict[str, float],
    ) -> "HybridBookRecommender":
        self.users_clean = users_clean
        self.books_clean = books_clean
        self.user_defaults = user_defaults
        self.book_defaults = book_defaults

        global_mean = float(ratings[RATING_COL].mean())
        user_bias, book_bias = fit_bias_terms(ratings, global_mean=global_mean)

        user_stats = ratings.groupby(USER_COL, observed=False)[RATING_COL].agg(["mean", "count"])
        book_stats = ratings.groupby(BOOK_COL, observed=False)[RATING_COL].agg(["mean", "count"])

        enriched = (
            ratings[[USER_COL, BOOK_COL, RATING_COL]]
            .merge(users_clean[[USER_COL, "Country", "Age_Bin"]], on=USER_COL, how="left")
            .merge(books_clean[[BOOK_COL, "Author", "Publisher", "Year_Bin"]], on=BOOK_COL, how="left")
        )
        enriched["Country"] = enriched["Country"].fillna("Unknown")
        enriched["Author"] = enriched["Author"].fillna("Unknown")
        enriched["Publisher"] = enriched["Publisher"].fillna("Unknown")
        enriched["Age_Bin"] = enriched["Age_Bin"].fillna(-1).astype(np.int16)
        enriched["Year_Bin"] = enriched["Year_Bin"].fillna(-1).astype(np.int16)

        author_mean, author_count = _smoothed_group_stats(enriched, "Author", global_mean)
        publisher_mean, publisher_count = _smoothed_group_stats(enriched, "Publisher", global_mean)
        country_mean, country_count = _smoothed_group_stats(enriched, "Country", global_mean)
        age_bin_mean, _ = _smoothed_group_stats(enriched, "Age_Bin", global_mean)
        year_bin_mean, _ = _smoothed_group_stats(enriched, "Year_Bin", global_mean)

        self.state = FeatureState(
            global_mean=global_mean,
            user_bias=user_bias,
            book_bias=book_bias,
            user_mean=user_stats["mean"].astype(np.float32),
            user_count=user_stats["count"].astype(np.float32),
            book_mean=book_stats["mean"].astype(np.float32),
            book_count=book_stats["count"].astype(np.float32),
            author_mean=author_mean,
            author_count=author_count,
            publisher_mean=publisher_mean,
            publisher_count=publisher_count,
            country_mean=country_mean,
            country_count=country_count,
            age_bin_mean=age_bin_mean,
            year_bin_mean=year_bin_mean,
        )

        x_train = self._build_features(ratings[[USER_COL, BOOK_COL]])
        bias_pred = self.predict_bias_only(ratings[[USER_COL, BOOK_COL]])
        residual = ratings[RATING_COL].to_numpy(dtype=np.float32) - bias_pred.astype(np.float32)
        self.model.fit(x_train, residual)
        return self

    def _build_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.state is None or self.users_clean is None or self.books_clean is None:
            raise RuntimeError("Model is not fitted.")

        frame = (
            pairs[[USER_COL, BOOK_COL]]
            .merge(self.users_clean, on=USER_COL, how="left")
            .merge(self.books_clean, on=BOOK_COL, how="left")
        )

        frame["Country"] = frame["Country"].fillna("Unknown")
        frame["Author"] = frame["Author"].fillna("Unknown")
        frame["Publisher"] = frame["Publisher"].fillna("Unknown")

        age_median = self.user_defaults["age_median"]
        year_median = self.book_defaults["year_median"]
        frame["Age_Clean"] = frame["Age_Clean"].fillna(age_median).astype(np.float32)
        frame["Age_Missing"] = frame["Age_Missing"].fillna(1).astype(np.float32)
        frame["Age_Bin"] = frame["Age_Bin"].fillna(-1).astype(np.int16)

        frame["Year_Clean"] = frame["Year_Clean"].fillna(year_median).astype(np.float32)
        frame["Year_Missing"] = frame["Year_Missing"].fillna(1).astype(np.float32)
        frame["Year_Bin"] = frame["Year_Bin"].fillna(-1).astype(np.int16)

        user_bias = frame[USER_COL].map(self.state.user_bias).fillna(0.0).astype(np.float32)
        book_bias = frame[BOOK_COL].map(self.state.book_bias).fillna(0.0).astype(np.float32)

        user_mean = frame[USER_COL].map(self.state.user_mean).fillna(self.state.global_mean).astype(np.float32)
        user_count = frame[USER_COL].map(self.state.user_count).fillna(0.0).astype(np.float32)
        book_mean = frame[BOOK_COL].map(self.state.book_mean).fillna(self.state.global_mean).astype(np.float32)
        book_count = frame[BOOK_COL].map(self.state.book_count).fillna(0.0).astype(np.float32)

        author_mean = frame["Author"].map(self.state.author_mean).fillna(self.state.global_mean).astype(np.float32)
        author_count = frame["Author"].map(self.state.author_count).fillna(0.0).astype(np.float32)
        publisher_mean = frame["Publisher"].map(self.state.publisher_mean).fillna(self.state.global_mean).astype(np.float32)
        publisher_count = frame["Publisher"].map(self.state.publisher_count).fillna(0.0).astype(np.float32)

        country_mean = frame["Country"].map(self.state.country_mean).fillna(self.state.global_mean).astype(np.float32)
        country_count = frame["Country"].map(self.state.country_count).fillna(0.0).astype(np.float32)

        age_bin_mean = frame["Age_Bin"].map(self.state.age_bin_mean).fillna(self.state.global_mean).astype(np.float32)
        year_bin_mean = frame["Year_Bin"].map(self.state.year_bin_mean).fillna(self.state.global_mean).astype(np.float32)

        cf_pred = (self.state.global_mean + user_bias + book_bias).astype(np.float32)
        user_count_log = np.log1p(user_count).astype(np.float32)
        book_count_log = np.log1p(book_count).astype(np.float32)

        x = pd.DataFrame(
            {
                "cf_pred": cf_pred,
                "user_mean": user_mean,
                "user_count_log": user_count_log,
                "book_mean": book_mean,
                "book_count_log": book_count_log,
                "author_mean": author_mean,
                "author_count_log": np.log1p(author_count).astype(np.float32),
                "publisher_mean": publisher_mean,
                "publisher_count_log": np.log1p(publisher_count).astype(np.float32),
                "country_mean": country_mean,
                "country_count_log": np.log1p(country_count).astype(np.float32),
                "age_bin_mean": age_bin_mean,
                "year_bin_mean": year_bin_mean,
                "age_clean": frame["Age_Clean"].astype(np.float32),
                "age_missing": frame["Age_Missing"].astype(np.float32),
                "year_clean": frame["Year_Clean"].astype(np.float32),
                "year_missing": frame["Year_Missing"].astype(np.float32),
                "cold_user": (user_count == 0.0).astype(np.float32),
                "cold_book": (book_count == 0.0).astype(np.float32),
                "activity_x_popularity": (user_count_log * book_count_log).astype(np.float32),
            }
        )
        return x

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        x = self._build_features(pairs)
        bias_pred = self.predict_bias_only(pairs)
        residual = self.model.predict(x)
        pred = bias_pred + residual
        return np.clip(pred, 0.0, 10.0)

    def predict_bias_only(self, pairs: pd.DataFrame) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Model is not fitted.")
        user_bias = pairs[USER_COL].map(self.state.user_bias).fillna(0.0).to_numpy()
        book_bias = pairs[BOOK_COL].map(self.state.book_bias).fillna(0.0).to_numpy()
        pred = self.state.global_mean + user_bias + book_bias
        return np.clip(pred, 0.0, 10.0)


def build_data_analysis_report(
    ratings_train: pd.DataFrame,
    ratings_test: pd.DataFrame,
    users: pd.DataFrame,
    books: pd.DataFrame,
    metrics: Dict[str, float],
    selected_model: str,
) -> str:
    train_users = set(ratings_train[USER_COL])
    train_books = set(ratings_train[BOOK_COL])
    cold_user_pct = float((~ratings_test[USER_COL].isin(train_users)).mean() * 100.0)
    cold_book_pct = float((~ratings_test[BOOK_COL].isin(train_books)).mean() * 100.0)
    zero_pct = float((ratings_train[RATING_COL] == 0).mean() * 100.0)

    age_invalid_pct = float((~users["Age"].between(5, 100)).mean() * 100.0)
    year = pd.to_numeric(books["Year"], errors="coerce")
    year_invalid_pct = float((~year.between(1800, 2026)).mean() * 100.0)

    ratings_per_user = ratings_train.groupby(USER_COL, observed=False).size()
    ratings_per_book = ratings_train.groupby(BOOK_COL, observed=False).size()

    report = f"""# Data Analysis Summary

## Dataset shape
- `ratings_train.csv`: {ratings_train.shape[0]:,} rows, {ratings_train.shape[1]} columns
- `ratings_to_predict.csv`: {ratings_test.shape[0]:,} rows, {ratings_test.shape[1]} columns
- `users.csv`: {users.shape[0]:,} rows, {users.shape[1]} columns
- `books.csv`: {books.shape[0]:,} rows, {books.shape[1]} columns

## What stands out
- The interaction matrix is extremely sparse (`~99.9966%` missing user-book cells).
- Ratings are very skewed toward zero ({zero_pct:.2f}% of train ratings are `0`).
- Cold-start exists in test:
  - unseen users: {cold_user_pct:.2f}% of test rows
  - unseen books: {cold_book_pct:.2f}% of test rows
- User activity is long-tailed:
  - median ratings/user: {ratings_per_user.median():.0f}
  - 99th percentile ratings/user: {ratings_per_user.quantile(0.99):.0f}
- Book popularity is long-tailed:
  - median ratings/book: {ratings_per_book.median():.0f}
  - 99th percentile ratings/book: {ratings_per_book.quantile(0.99):.0f}

## Cleaning / normalization decisions
- `Age`: values outside `[5, 100]` treated as invalid ({age_invalid_pct:.2f}%); imputed with median and flagged with `Age_Missing`.
- `Year`: values outside `[1800, 2026]` treated as invalid ({year_invalid_pct:.2f}%); imputed with median and flagged with `Year_Missing`.
- Categorical missing/blank values (`Country`, `Author`, `Publisher`) filled with `Unknown`.
- No z-score normalization was needed for tree-based final model.
- Instead of one-hot encoding huge categorical spaces, smoothed target statistics were used (`author_mean`, `publisher_mean`, `country_mean`, etc.).

## Modeling approach
- Collaborative filtering baseline: regularized user/book bias model.
- Hybrid model: gradient boosting over CF prediction + user/book stats + metadata stats.
- Validation RMSE:
  - global mean baseline: {metrics["rmse_global"]:.5f}
  - bias-only CF: {metrics["rmse_bias"]:.5f}
  - hybrid model: {metrics["rmse_hybrid"]:.5f}
- Selected model for submission: `{selected_model}` (lowest validation RMSE).
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hybrid recommender and generate Kaggle submission.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    ratings_train = pd.read_csv(args.data_dir / "ratings_train.csv")
    ratings_test = pd.read_csv(args.data_dir / "ratings_to_predict.csv")
    users_raw = pd.read_csv(args.data_dir / "users.csv")
    books_raw = pd.read_csv(args.data_dir / "books.csv")

    users_clean, user_defaults = clean_users(users_raw)
    books_clean, book_defaults = clean_books(books_raw)

    train_split, val_split = train_test_split(
        ratings_train[[USER_COL, BOOK_COL, RATING_COL]],
        test_size=args.test_size,
        random_state=args.random_state,
    )

    validation_model = HybridBookRecommender(random_state=args.random_state)
    validation_model.fit(train_split, users_clean, books_clean, user_defaults, book_defaults)

    y_val = val_split[RATING_COL]
    pred_global = np.full(shape=len(val_split), fill_value=validation_model.state.global_mean, dtype=np.float32)
    pred_bias = validation_model.predict_bias_only(val_split[[USER_COL, BOOK_COL]])
    pred_hybrid = validation_model.predict(val_split[[USER_COL, BOOK_COL]])

    metrics = {
        "rmse_global": rmse(y_val, pred_global),
        "rmse_bias": rmse(y_val, pred_bias),
        "rmse_hybrid": rmse(y_val, pred_hybrid),
    }
    selected_model = "bias" if metrics["rmse_bias"] <= metrics["rmse_hybrid"] else "hybrid"

    final_model = HybridBookRecommender(random_state=args.random_state)
    final_model.fit(
        ratings_train[[USER_COL, BOOK_COL, RATING_COL]],
        users_clean,
        books_clean,
        user_defaults,
        book_defaults,
    )

    if selected_model == "bias":
        pred_test = final_model.predict_bias_only(ratings_test[[USER_COL, BOOK_COL]])
    else:
        pred_test = final_model.predict(ratings_test[[USER_COL, BOOK_COL]])
    pred_test_int = np.rint(pred_test).astype(int)
    pred_test_int = np.clip(pred_test_int, 0, 10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            # canonical sample-submission keys 0..n-1
            "Id": np.arange(len(ratings_test), dtype=int),
            "Rating": pred_test_int,
        }
    )
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    metrics_path = args.output_dir / "validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = build_data_analysis_report(
        ratings_train,
        ratings_test,
        users_raw,
        books_raw,
        metrics,
        selected_model=selected_model,
    )
    report_path = args.output_dir / "data_analysis.md"
    report_path.write_text(report, encoding="utf-8")

    print("Validation RMSE:")
    print(f"  global baseline: {metrics['rmse_global']:.5f}")
    print(f"  bias baseline  : {metrics['rmse_bias']:.5f}")
    print(f"  hybrid model   : {metrics['rmse_hybrid']:.5f}")
    print(f"  selected model : {selected_model}")
    print()
    print(f"Submission file written to: {submission_path}")
    print(f"Validation metrics written to: {metrics_path}")
    print(f"Data analysis report written to: {report_path}")


if __name__ == "__main__":
    main()
