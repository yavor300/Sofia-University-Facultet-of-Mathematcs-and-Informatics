from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

from ..common import option_level_frames, paired_predictions_from_scores
from .base import EnabledModelConfig, PredictionMap


@dataclass
class SvmPairConfig(EnabledModelConfig):
    name: str = "svm_pair"
    max_features: int = 50000
    c_value: float = 1.0


def run(train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg: SvmPairConfig) -> PredictionMap:
    train_opt, dev_opt = option_level_frames(train_df, dev_df)
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=cfg.max_features, ngram_range=(1, 2), lowercase=True)),
            ("svm", LinearSVC(C=cfg.c_value)),
        ]
    )
    with tqdm(total=2, desc="SVM train+test", leave=False) as pbar:
        pipe.fit(train_opt["feature_text"], train_opt["target"])
        pbar.update(1)
        scores = pipe.decision_function(dev_opt["feature_text"])
        pbar.update(1)
    return paired_predictions_from_scores(dev_opt, np.asarray(scores))
