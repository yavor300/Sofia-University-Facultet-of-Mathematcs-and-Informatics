from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...modeling import TransformerOptionPairClassifier
from .base import EnabledModelConfig, PredictionMap


@dataclass
class TransformerCrossEncoderConfig(EnabledModelConfig):
    name: str = "transformer_cross_encoder"
    model_dir: str = "models/option_pair_transformer_finmmeval"


def run(train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg: TransformerCrossEncoderConfig) -> PredictionMap:
    del train_df
    model_path = Path(cfg.model_dir)
    if not model_path.exists():
        return {}
    model = TransformerOptionPairClassifier.load(model_path)
    return model.predict(dev_df)

