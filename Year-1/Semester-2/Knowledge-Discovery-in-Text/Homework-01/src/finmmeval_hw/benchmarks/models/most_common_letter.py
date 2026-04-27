from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from .base import EnabledModelConfig, PredictionMap


@dataclass
class MostCommonLetterConfig(EnabledModelConfig):
    name: str = "most_common_letter_baseline"


def run(train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg: MostCommonLetterConfig) -> PredictionMap:
    letters = [row[0] for row in train_df["gold_letters"] if row]
    common = pd.Series(letters).value_counts().idxmax() if letters else "a"
    predictions: PredictionMap = {}
    for qid in tqdm(dev_df["id"].tolist(), desc="Most-common baseline", leave=False):
        predictions[str(qid)] = [str(common)]
    return predictions
