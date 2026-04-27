from __future__ import annotations

from .benchmarks.runner import main, run_with_default_config


def run_extended_benchmarks(
    input_jsonl: str,
    output_json: str,
    output_md: str,
    seed: int = 42,
    dev_size: float = 0.2,
    transformer_model_dir: str = "models/option_pair_transformer_finmmeval",
) -> dict:
    """Backward-compatible wrapper around the structured benchmark runner."""
    return run_with_default_config(
        input_jsonl=input_jsonl,
        output_json=output_json,
        output_md=output_md,
        seed=seed,
        dev_size=dev_size,
        transformer_model_dir=transformer_model_dir,
    )


if __name__ == "__main__":
    main()

