# RunPod Llama 3 QLoRA Experiment

This guide trains `meta-llama/Meta-Llama-3-8B-Instruct` on the combined English
CPA/CFA + BhashaBench-Finance MCQ data.

## 1. Copy The Project To RunPod

Run this from the repository root on your local machine:

```bash
rsync -avz \
  -e "ssh -i ~/.ssh/id_ed25519_runpod" \
  Year-1/Semester-2/Knowledge-Discovery-in-Text/Homework-01/ \
  4ggeomqs0yqy7z-6441163c@ssh.runpod.io:~/Homework-01/
```

If you already prepared the combined JSONL locally, this copies it too:

```text
data/processed/english_questions_combined.jsonl
```

## 2. Connect To RunPod

```bash
ssh 4ggeomqs0yqy7z-6441163c@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod
cd ~/Homework-01
```

## 3. Install Dependencies

Use a CUDA/PyTorch RunPod image if possible.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-runpod-llama.txt
```

Check GPU visibility:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 4. Authenticate Hugging Face

Both Llama 3 and BBF may require Hugging Face access.

```bash
huggingface-cli login
```

You need to accept the Llama 3 license on Hugging Face for the same account.

## 5. Prepare Combined Data If Needed

Skip this if `data/processed/english_questions_combined.jsonl` was copied.

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli prepare \
  --source both \
  --bbf-language English \
  --bbf-split test \
  --bbf-question-type MCQ \
  --bbf-use-token \
  --output data/processed/english_questions_combined.jsonl
```

## 6. Train Llama 3 On 100% Of Combined Data

The default config uses `sample_ratio: 1.0`, which means 100% of the combined
dataset is used before the train/dev split.

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.llama_qlora \
  --config configs/llama_qlora.yaml
```

Equivalent launcher:

```bash
bash scripts/run_llama_qlora.sh
```

## 7. Outputs

The default config writes:

```text
models/llama3_8b_qlora_combined/
results/llama3_8b_qlora_combined_metrics.json
results/llama3_8b_qlora_combined_predictions.csv
results/llama3_8b_qlora_combined_split.json
```

## 8. Tune The Run

Edit:

```text
configs/llama_qlora.yaml
```

Useful settings:

```yaml
training:
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0002

lora:
  r: 16
  alpha: 32
```

For a smaller test run before the full experiment:

```yaml
data:
  sample_ratio: 0.1
```
