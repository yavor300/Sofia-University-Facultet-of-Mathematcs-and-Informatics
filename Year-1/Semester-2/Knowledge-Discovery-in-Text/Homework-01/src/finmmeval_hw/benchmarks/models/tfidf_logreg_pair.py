from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...modeling import OptionPairClassifier
from .base import EnabledModelConfig, PredictionMap


@dataclass
class TfidfLogregPairConfig(EnabledModelConfig):
    name: str = "tfidf_logreg_pair"
    max_features: int = 40000
    ngram_min: int = 1
    ngram_max: int = 2
    c_value: float = 1.0


def run(train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg: TfidfLogregPairConfig) -> PredictionMap:
    model = OptionPairClassifier(
        max_features=cfg.max_features,
        ngram_min=cfg.ngram_min,
        ngram_max=cfg.ngram_max,
        c_value=cfg.c_value,
    )
    model.fit(train_df)
    return model.predict(dev_df)

