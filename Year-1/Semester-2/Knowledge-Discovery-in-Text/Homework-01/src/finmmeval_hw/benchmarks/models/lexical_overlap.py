from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...modeling import LexicalOverlapRanker
from .base import EnabledModelConfig, PredictionMap


@dataclass
class LexicalOverlapConfig(EnabledModelConfig):
    name: str = "lexical_overlap"


def run(train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg: LexicalOverlapConfig) -> PredictionMap:
    del train_df
    return LexicalOverlapRanker().predict(dev_df)

