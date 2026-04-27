from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ..data import load_questions_jsonl
from ..evaluation import evaluate_predictions
from .common import filter_4d_single_answer, make_train_dev_split
from .experiment_config import BenchmarkConfig, load_benchmark_config
from .models import lexical_overlap, mlp_pair, most_common_letter, multiclass_4d_svm_summary
from .models import svm_pair, tfidf_logreg_pair, transformer_cross_encoder, word2vec_mlp_pair
from tqdm import tqdm


def _sample_questions(questions_df, sample_ratio: float, seed: int):
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got: {sample_ratio}")
    if sample_ratio >= 1.0:
        return questions_df.reset_index(drop=True)

    sample_size = max(1, int(len(questions_df) * sample_ratio))
    sampled = questions_df.sample(n=sample_size, random_state=seed, replace=False)
    return sampled.reset_index(drop=True)


def run_benchmarks(cfg: BenchmarkConfig) -> dict:
    all_questions_df = load_questions_jsonl(cfg.io.input_jsonl)
    questions_df = _sample_questions(
        questions_df=all_questions_df,
        sample_ratio=cfg.split.sample_ratio,
        seed=cfg.split.seed,
    )
    train_df, dev_df = make_train_dev_split(
        questions_df=questions_df,
        dev_size=cfg.split.dev_size,
        seed=cfg.split.seed,
    )

    results = {
        "loaded_size": len(all_questions_df),
        "sampled_size": len(questions_df),
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "seed": cfg.split.seed,
        "dev_size_ratio": cfg.split.dev_size,
        "sample_ratio": cfg.split.sample_ratio,
        "models": {},
    }

    model_jobs = [
        ("most_common_letter_baseline", cfg.models.most_common_letter, most_common_letter.run),
        ("lexical_overlap", cfg.models.lexical_overlap, lexical_overlap.run),
        ("tfidf_logreg_pair", cfg.models.tfidf_logreg_pair, tfidf_logreg_pair.run),
        ("svm_pair", cfg.models.svm_pair, svm_pair.run),
        ("mlp_pair", cfg.models.mlp_pair, lambda tr, dv, c: mlp_pair.run(tr, dv, c, seed=cfg.split.seed)),
        (
            "word2vec_mlp_pair",
            cfg.models.word2vec_mlp_pair,
            lambda tr, dv, c: word2vec_mlp_pair.run(tr, dv, c, seed=cfg.split.seed),
        ),
        (
            "multiclass_4d_svm_summary",
            cfg.models.multiclass_4d_svm_summary,
            multiclass_4d_svm_summary.run,
        ),
        ("transformer_cross_encoder", cfg.models.transformer_cross_encoder, transformer_cross_encoder.run),
    ]

    progress = tqdm(model_jobs, desc="Benchmark models", leave=True)
    for metric_name, model_cfg, runner in progress:
        progress.set_postfix_str(metric_name)
        if not getattr(model_cfg, "enabled", True):
            continue
        preds = runner(train_df, dev_df, model_cfg)
        if metric_name == "multiclass_4d_svm_summary":
            target_df = filter_4d_single_answer(dev_df)
            if not preds:
                results["models"][metric_name] = {
                    "evaluated_questions": 0,
                    "exact_match_accuracy": 0.0,
                    "top1_accuracy": 0.0,
                }
                results["multiclass_4d_dev_questions"] = 0
                continue
            results["multiclass_4d_dev_questions"] = len(target_df)
            results["models"][metric_name] = evaluate_predictions(target_df, preds)
            continue
        if not preds:
            continue
        results["models"][metric_name] = evaluate_predictions(dev_df, preds)

    if "multiclass_4d_dev_questions" not in results:
        results["multiclass_4d_dev_questions"] = len(filter_4d_single_answer(dev_df))

    return results


def write_outputs(results: dict, output_json: str, output_md: str) -> None:
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    rows: List[tuple] = []
    for model_name, metrics in results["models"].items():
        rows.append((model_name, metrics["exact_match_accuracy"], metrics["top1_accuracy"]))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)

    md_lines = [
        f"# Extended Dev Comparison (seed={results['seed']}, dev={results['dev_size_ratio']})",
        "",
        "| Model | Exact Match | Top-1 |",
        "|---|---:|---:|",
    ]
    for name, exact, top1 in rows:
        md_lines.append(f"| {name} | {exact:.4f} | {top1:.4f} |")
    md_lines.append("")
    md_lines.append(f"- Loaded size: {results['loaded_size']}")
    md_lines.append(f"- Sample ratio: {results['sample_ratio']}")
    md_lines.append(f"- Sampled size: {results['sampled_size']}")
    md_lines.append(f"- Train size: {results['train_size']}")
    md_lines.append(f"- Dev size: {results['dev_size']}")
    md_lines.append(f"- 4D multiclass dev subset size: {results['multiclass_4d_dev_questions']}")

    md_path = Path(output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def run_with_default_config(
    config_yaml: str | None = None,
    input_jsonl: str | None = None,
    output_json: str | None = None,
    output_md: str | None = None,
    seed: int | None = None,
    dev_size: float | None = None,
    sample_ratio: float | None = None,
    transformer_model_dir: str | None = None,
) -> dict:
    cfg = load_benchmark_config(config_yaml=config_yaml)
    if input_jsonl is not None:
        cfg.io.input_jsonl = input_jsonl
    if output_json is not None:
        cfg.io.output_json = output_json
    if output_md is not None:
        cfg.io.output_md = output_md
    if seed is not None:
        cfg.split.seed = seed
    if dev_size is not None:
        cfg.split.dev_size = dev_size
    if sample_ratio is not None:
        cfg.split.sample_ratio = sample_ratio
    if transformer_model_dir is not None:
        cfg.models.transformer_cross_encoder.model_dir = transformer_model_dir

    results = run_benchmarks(cfg)
    write_outputs(results, output_json=cfg.io.output_json, output_md=cfg.io.output_md)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured extended benchmarks.")
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Path to a benchmark YAML config file.",
    )
    parser.add_argument("--input-jsonl", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dev-size", type=float, default=None)
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Keep only this percentage of input rows before splitting (for example, 0.1).",
    )
    parser.add_argument("--transformer-model-dir", type=str, default=None)
    args = parser.parse_args()

    results = run_with_default_config(
        config_yaml=args.config_yaml,
        input_jsonl=args.input_jsonl,
        output_json=args.output_json,
        output_md=args.output_md,
        seed=args.seed,
        dev_size=args.dev_size,
        sample_ratio=args.sample_ratio,
        transformer_model_dir=args.transformer_model_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
