from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from ..common import TOKEN_RE, option_level_frames, paired_predictions_from_scores
from .base import EnabledModelConfig, PredictionMap


@dataclass
class Word2VecMlpPairConfig(EnabledModelConfig):
    name: str = "word2vec_mlp_pair"
    vector_size: int = 200
    window: int = 5
    epochs: int = 20
    hidden_layer_sizes: tuple = (256, 64)
    max_iter: int = 300


def _mean_embedding(tokens: List[str], wv, dim: int) -> np.ndarray:
    vecs = [wv[token] for token in tokens if token in wv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.vstack(vecs), axis=0)


def run(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    cfg: Word2VecMlpPairConfig,
    seed: int,
) -> PredictionMap:
    try:
        from gensim.models import Word2Vec
    except Exception as exc:
        raise RuntimeError("`gensim` is required for Word2Vec benchmark.") from exc

    train_opt, dev_opt = option_level_frames(train_df, dev_df)

    corpus = []
    for col in ["question", "option_text"]:
        texts = train_opt[col].astype(str).tolist()
        for text in tqdm(texts, desc=f"Word2Vec corpus ({col})", leave=False):
            corpus.append([t.lower() for t in TOKEN_RE.findall(text)])

    w2v = Word2Vec(
        sentences=corpus,
        vector_size=cfg.vector_size,
        window=cfg.window,
        min_count=1,
        workers=1,
        sg=1,
        epochs=cfg.epochs,
        seed=seed,
    )
    dim = w2v.vector_size

    def build_features(frame: pd.DataFrame) -> np.ndarray:
        feats = []
        iterator = frame.iterrows()
        for _, row in tqdm(
            iterator,
            total=len(frame),
            desc="Build Word2Vec features",
            leave=False,
        ):
            q_tokens = [t.lower() for t in TOKEN_RE.findall(str(row["question"]))]
            o_tokens = [t.lower() for t in TOKEN_RE.findall(str(row["option_text"]))]
            q_vec = _mean_embedding(q_tokens, w2v.wv, dim)
            o_vec = _mean_embedding(o_tokens, w2v.wv, dim)
            feats.append(np.concatenate([q_vec, o_vec, np.abs(q_vec - o_vec), q_vec * o_vec]))
        return np.asarray(feats)

    x_train = build_features(train_opt)
    x_dev = build_features(dev_opt)
    y_train = train_opt["target"].astype(int).values

    clf = MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        activation="relu",
        learning_rate_init=5e-4,
        max_iter=cfg.max_iter,
        random_state=seed,
    )
    with tqdm(total=2, desc="Word2Vec+MLP train+test", leave=False) as pbar:
        clf.fit(x_train, y_train)
        pbar.update(1)
        scores = clf.predict_proba(x_dev)[:, 1]
        pbar.update(1)
    return paired_predictions_from_scores(dev_opt, np.asarray(scores))
