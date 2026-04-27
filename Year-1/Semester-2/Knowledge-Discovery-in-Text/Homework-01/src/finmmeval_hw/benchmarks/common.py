from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data import build_option_level_frame

SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")
TOKEN_RE = re.compile(r"[A-Za-z]{2,}")


def make_train_dev_split(
    questions_df: pd.DataFrame,
    dev_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eligible = questions_df[questions_df["gold_letters"].map(len) > 0].copy()
    stratify_labels = eligible["gold_letters"].map(lambda x: x[0] if x else "none")

    try:
        train_ids, dev_ids = train_test_split(
            eligible["id"],
            test_size=dev_size,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        train_ids, dev_ids = train_test_split(
            eligible["id"],
            test_size=dev_size,
            random_state=seed,
            shuffle=True,
        )

    train_df = questions_df[questions_df["id"].isin(train_ids)].reset_index(drop=True)
    dev_df = questions_df[questions_df["id"].isin(dev_ids)].reset_index(drop=True)
    return train_df, dev_df


def question_summary(question: str, max_sentences: int = 2) -> str:
    text = " ".join(str(question).split())
    if not text:
        return text
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return " ".join(sentences[:max_sentences]) if sentences else text


def paired_predictions_from_scores(option_df: pd.DataFrame, scores: np.ndarray) -> Dict[str, List[str]]:
    out = option_df[["id", "label"]].copy()
    out["score"] = scores
    predictions: Dict[str, List[str]] = {}
    for qid, group in out.groupby("id", sort=False):
        best = group.sort_values(["score", "label"], ascending=[False, True]).iloc[0]
        predictions[str(qid)] = [str(best["label"])]
    return predictions


def option_level_frames(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_opt = build_option_level_frame(train_df, with_targets=True)
    dev_opt = build_option_level_frame(dev_df, with_targets=False)
    return train_opt, dev_opt


def filter_4d_single_answer(df: pd.DataFrame) -> pd.DataFrame:
    def ok(row: pd.Series) -> bool:
        labels = row["choice_labels"]
        gold = row["gold_letters"]
        return labels == ["a", "b", "c", "d"] and len(gold) == 1 and gold[0] in {"a", "b", "c", "d"}

    return df[df.apply(ok, axis=1)].reset_index(drop=True)

