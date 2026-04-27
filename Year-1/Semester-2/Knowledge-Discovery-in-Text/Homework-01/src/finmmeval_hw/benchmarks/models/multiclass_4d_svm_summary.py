from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

from ..common import filter_4d_single_answer, question_summary
from .base import EnabledModelConfig, PredictionMap


@dataclass
class Multiclass4dSvmSummaryConfig(EnabledModelConfig):
    name: str = "multiclass_4d_svm_summary"
    max_features: int = 50000
    c_value: float = 1.0
    summary_max_sentences: int = 2


def _build_text(row: pd.Series, max_sentences: int) -> str:
    summary = question_summary(str(row["question"]), max_sentences=max_sentences)
    opts = row["option_texts"]
    return (
        f"summary: {summary}\n"
        f"option_a: {opts.get('a', '')}\n"
        f"option_b: {opts.get('b', '')}\n"
        f"option_c: {opts.get('c', '')}\n"
        f"option_d: {opts.get('d', '')}"
    )


def run(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    cfg: Multiclass4dSvmSummaryConfig,
) -> PredictionMap:
    train_4d = filter_4d_single_answer(train_df)
    dev_4d = filter_4d_single_answer(dev_df)
    if train_4d.empty or dev_4d.empty:
        return {}

    train_iter = train_4d.iterrows()
    x_train = [
        _build_text(row, max_sentences=cfg.summary_max_sentences)
        for _, row in tqdm(
            train_iter,
            total=len(train_4d),
            desc="Build 4D train text",
            leave=False,
        )
    ]
    y_train = train_4d["gold_letters"].map(lambda x: x[0]).tolist()
    dev_iter = dev_4d.iterrows()
    x_dev = [
        _build_text(row, max_sentences=cfg.summary_max_sentences)
        for _, row in tqdm(
            dev_iter,
            total=len(dev_4d),
            desc="Build 4D dev text",
            leave=False,
        )
    ]

    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=cfg.max_features, ngram_range=(1, 2), lowercase=True)),
            ("svm", LinearSVC(C=cfg.c_value)),
        ]
    )
    with tqdm(total=2, desc="4D SVM train+test", leave=False) as pbar:
        clf.fit(x_train, y_train)
        pbar.update(1)
        pred_labels = clf.predict(x_dev)
        pbar.update(1)
    return {str(qid): [str(label)] for qid, label in zip(dev_4d["id"].tolist(), pred_labels)}
