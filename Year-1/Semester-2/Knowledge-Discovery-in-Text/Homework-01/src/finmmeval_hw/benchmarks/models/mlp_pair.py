from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..common import option_level_frames, paired_predictions_from_scores
from .base import EnabledModelConfig, PredictionMap


@dataclass
class MlpPairConfig(EnabledModelConfig):
    name: str = "mlp_pair"
    max_features: int = 50000
    svd_components: int = 256
    hidden_layer_sizes: tuple = (256, 64)
    max_iter: int = 250


def run(train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg: MlpPairConfig, seed: int) -> PredictionMap:
    train_opt, dev_opt = option_level_frames(train_df, dev_df)
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=cfg.max_features, ngram_range=(1, 2), lowercase=True)),
            ("svd", TruncatedSVD(n_components=cfg.svd_components, random_state=seed)),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=cfg.hidden_layer_sizes,
                    activation="relu",
                    learning_rate_init=1e-3,
                    max_iter=cfg.max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    with tqdm(total=2, desc="MLP train+test", leave=False) as pbar:
        pipe.fit(train_opt["feature_text"], train_opt["target"])
        pbar.update(1)
        scores = pipe.predict_proba(dev_opt["feature_text"])[:, 1]
        pbar.update(1)
    return paired_predictions_from_scores(dev_opt, np.asarray(scores))
