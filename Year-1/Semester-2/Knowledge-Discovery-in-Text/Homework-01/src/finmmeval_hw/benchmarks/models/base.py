from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import pandas as pd


PredictionMap = Dict[str, List[str]]


class ModelRunner(Protocol):
    def __call__(self, train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg) -> PredictionMap:
        ...


@dataclass
class EnabledModelConfig:
    enabled: bool = True

