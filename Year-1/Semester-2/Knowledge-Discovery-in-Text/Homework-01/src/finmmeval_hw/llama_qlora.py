from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .data import load_questions_jsonl
from .evaluation import evaluate_predictions

ANSWER_RE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _sample_questions(df: pd.DataFrame, sample_ratio: float, seed: int) -> pd.DataFrame:
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got: {sample_ratio}")
    if sample_ratio >= 1.0:
        return df.reset_index(drop=True)
    sample_size = max(1, int(len(df) * sample_ratio))
    return df.sample(n=sample_size, random_state=seed, replace=False).reset_index(drop=True)


def _make_split(df: pd.DataFrame, dev_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df[df["gold_letters"].map(len) > 0].copy()
    if labeled.empty:
        raise ValueError("No labeled examples found.")

    if dev_size <= 0:
        return labeled.reset_index(drop=True), labeled.iloc[0:0].copy()

    stratify = labeled["gold_letters"].map(lambda x: x[0] if x else "none")
    try:
        train_ids, dev_ids = train_test_split(
            labeled["id"],
            test_size=dev_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_ids, dev_ids = train_test_split(
            labeled["id"],
            test_size=dev_size,
            random_state=seed,
            shuffle=True,
        )
    train_df = labeled[labeled["id"].isin(train_ids)].reset_index(drop=True)
    dev_df = labeled[labeled["id"].isin(dev_ids)].reset_index(drop=True)
    return train_df, dev_df


def _format_options(row: pd.Series) -> str:
    lines = []
    for label in row["choice_labels"]:
        text = str(row["option_texts"].get(label, "")).strip()
        lines.append(f"{label.upper()}. {text}")
    return "\n".join(lines)


def build_prompt(row: pd.Series) -> str:
    return (
        "You are answering a financial multiple-choice question.\n"
        "Return only the correct option letter. If multiple options are correct, "
        "return the letters separated by commas.\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Options:\n{_format_options(row)}\n\n"
        "Answer:"
    )


def build_answer(row: pd.Series) -> str:
    return " " + ", ".join(str(label).upper() for label in row["gold_letters"])


def _tokenize_row(row: dict, tokenizer, max_length: int) -> dict:
    prompt = row["prompt"]
    answer = row["answer"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        prompt + answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    labels = list(full["input_ids"])
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    full["labels"] = labels
    return full


def _rows_to_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    from datasets import Dataset

    records = [
        {
            "id": str(row["id"]),
            "prompt": build_prompt(row),
            "answer": build_answer(row),
        }
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Build Llama prompts")
    ]
    dataset = Dataset.from_list(records)
    return dataset.map(
        lambda row: _tokenize_row(row, tokenizer=tokenizer, max_length=max_length),
        remove_columns=dataset.column_names,
        desc="Tokenize Llama dataset",
    )


def _quantization_config(cfg: dict):
    from transformers import BitsAndBytesConfig

    model_cfg = cfg["model"]
    if not model_cfg.get("load_in_4bit", True):
        return None
    dtype_name = str(model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    compute_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _load_model_and_tokenizer(cfg: dict):
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Llama QLoRA dependencies are missing. Install `requirements-runpod-llama.txt`."
        ) from exc

    model_name = cfg["model"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_quantization_config(cfg),
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg["training"].get("bf16", True) else torch.float16,
    )
    if cfg["training"].get("gradient_checkpointing", True):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model, tokenizer


def _extract_prediction(text: str, allowed_labels: List[str]) -> List[str]:
    allowed = {label.lower() for label in allowed_labels}
    found = []
    for match in ANSWER_RE.findall(text):
        label = match.lower()
        if label in allowed and label not in found:
            found.append(label)
    return found[: max(1, len(found))] if found else []


def _predict_dev(model, tokenizer, dev_df: pd.DataFrame, cfg: dict) -> Dict[str, List[str]]:
    predictions: Dict[str, List[str]] = {}
    model.eval()
    max_new_tokens = int(cfg["generation"].get("max_new_tokens", 8))
    do_sample = bool(cfg["generation"].get("do_sample", False))

    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="Llama dev inference"):
        prompt = build_prompt(row)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0][encoded["input_ids"].shape[1] :]
        answer_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        predictions[str(row["id"])] = _extract_prediction(answer_text, row["choice_labels"])
    return predictions


def train_from_config(config_path: str | Path) -> dict:
    cfg = _load_yaml(config_path)
    data_cfg = cfg["data"]
    outputs_cfg = cfg["outputs"]

    full_df = load_questions_jsonl(data_cfg["input_jsonl"])
    sampled_df = _sample_questions(
        full_df,
        sample_ratio=float(data_cfg.get("sample_ratio", 1.0)),
        seed=int(data_cfg.get("seed", 42)),
    )
    train_df, dev_df = _make_split(
        sampled_df,
        dev_size=float(data_cfg.get("dev_size", 0.2)),
        seed=int(data_cfg.get("seed", 42)),
    )

    Path(cfg["model"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    for key in ("metrics_json", "predictions_csv", "split_json"):
        Path(outputs_cfg[key]).parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_and_tokenizer(cfg)
    train_dataset = _rows_to_dataset(train_df, tokenizer, max_length=int(cfg["model"]["max_length"]))
    eval_dataset = (
        _rows_to_dataset(dev_df, tokenizer, max_length=int(cfg["model"]["max_length"]))
        if not dev_df.empty
        else None
    )

    train_cfg = cfg["training"]
    try:
        from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError(
            "Llama QLoRA dependencies are missing. Install `requirements-runpod-llama.txt`."
        ) from exc

    training_args = TrainingArguments(
        output_dir=cfg["model"]["output_dir"],
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        bf16=bool(train_cfg["bf16"]),
        fp16=bool(train_cfg["fp16"]),
        report_to=str(train_cfg.get("report_to", "none")),
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
    )
    trainer.train()
    trainer.save_model(cfg["model"]["output_dir"])
    tokenizer.save_pretrained(cfg["model"]["output_dir"])

    predictions = _predict_dev(model, tokenizer, dev_df, cfg) if not dev_df.empty else {}
    metrics = evaluate_predictions(dev_df, predictions) if predictions else {
        "evaluated_questions": 0,
        "exact_match_accuracy": 0.0,
        "top1_accuracy": 0.0,
    }
    payload = {
        "model_name": cfg["model"]["model_name"],
        "loaded_size": len(full_df),
        "sampled_size": len(sampled_df),
        "train_size": len(train_df),
        "dev_size": len(dev_df),
        "sample_ratio": float(data_cfg.get("sample_ratio", 1.0)),
        "seed": int(data_cfg.get("seed", 42)),
        "dev_size_ratio": float(data_cfg.get("dev_size", 0.2)),
        "metrics": metrics,
    }
    Path(outputs_cfg["metrics_json"]).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    Path(outputs_cfg["split_json"]).write_text(
        json.dumps(
            {
                "seed": payload["seed"],
                "dev_size": payload["dev_size_ratio"],
                "sample_ratio": payload["sample_ratio"],
                "train_ids": train_df["id"].astype(str).tolist(),
                "dev_ids": dev_df["id"].astype(str).tolist(),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"id": qid, "prediction": ",".join(labels)} for qid, labels in predictions.items()]
    ).to_csv(outputs_cfg["predictions_csv"], index=False)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 with QLoRA.")
    parser.add_argument("--config", type=str, default="configs/llama_qlora.yaml")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
