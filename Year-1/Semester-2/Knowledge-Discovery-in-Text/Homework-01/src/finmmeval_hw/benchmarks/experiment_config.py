from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .models.lexical_overlap import LexicalOverlapConfig
from .models.mlp_pair import MlpPairConfig
from .models.most_common_letter import MostCommonLetterConfig
from .models.multiclass_4d_svm_summary import Multiclass4dSvmSummaryConfig
from .models.svm_pair import SvmPairConfig
from .models.tfidf_logreg_pair import TfidfLogregPairConfig
from .models.transformer_cross_encoder import TransformerCrossEncoderConfig
from .models.word2vec_mlp_pair import Word2VecMlpPairConfig


@dataclass
class BenchmarkIOConfig:
    input_jsonl: str = "data/processed/english_questions_finmmeval.jsonl"
    output_json: str = "results/extended_benchmarks_finmmeval.json"
    output_md: str = "results/extended_benchmarks_finmmeval.md"


@dataclass
class BenchmarkSplitConfig:
    seed: int = 42
    dev_size: float = 0.2
    sample_ratio: float = 1.0


@dataclass
class BenchmarkModelConfig:
    most_common_letter: MostCommonLetterConfig = field(default_factory=MostCommonLetterConfig)
    lexical_overlap: LexicalOverlapConfig = field(default_factory=LexicalOverlapConfig)
    tfidf_logreg_pair: TfidfLogregPairConfig = field(default_factory=TfidfLogregPairConfig)
    svm_pair: SvmPairConfig = field(default_factory=SvmPairConfig)
    mlp_pair: MlpPairConfig = field(default_factory=MlpPairConfig)
    word2vec_mlp_pair: Word2VecMlpPairConfig = field(default_factory=Word2VecMlpPairConfig)
    multiclass_4d_svm_summary: Multiclass4dSvmSummaryConfig = field(
        default_factory=Multiclass4dSvmSummaryConfig
    )
    transformer_cross_encoder: TransformerCrossEncoderConfig = field(
        default_factory=TransformerCrossEncoderConfig
    )


@dataclass
class BenchmarkConfig:
    io: BenchmarkIOConfig = field(default_factory=BenchmarkIOConfig)
    split: BenchmarkSplitConfig = field(default_factory=BenchmarkSplitConfig)
    models: BenchmarkModelConfig = field(default_factory=BenchmarkModelConfig)


def default_benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig()


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def _apply_updates(target: Any, updates: dict, path: str = "") -> None:
    if not is_dataclass(target):
        raise TypeError("Target must be a dataclass instance.")
    if not isinstance(updates, dict):
        raise ValueError(f"Expected mapping at `{path or 'root'}`, got: {type(updates).__name__}")

    allowed = {f.name for f in fields(target)}
    for key, value in updates.items():
        key_path = f"{path}.{key}" if path else key
        if key not in allowed:
            raise ValueError(f"Unknown config key: `{key_path}`")
        current = getattr(target, key)
        if is_dataclass(current):
            _apply_updates(current, value, path=key_path)
            continue
        setattr(target, key, _coerce_value(current, value))


def load_benchmark_config(config_yaml: str | Path | None = None) -> BenchmarkConfig:
    cfg = default_benchmark_config()
    if config_yaml is None:
        return cfg

    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for --config-yaml support.") from exc

    path = Path(config_yaml)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping in: {path}")
    _apply_updates(cfg, payload)
    return cfg
