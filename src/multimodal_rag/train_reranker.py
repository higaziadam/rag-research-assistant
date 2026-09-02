from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


def _load_rows(data_path: str) -> List[Dict[str, Any]]:
    path = Path(data_path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("Training data is empty.")

    try:
        parsed = json.loads(raw)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]

    examples = []
    for row in rows:
        if not {"query", "passage", "label"}.issubset(row):
            continue
        label = float(row["label"])
        if label not in (0.0, 1.0):
            raise ValueError("Reranker labels must be 0 or 1.")
        examples.append({"query": str(row["query"]), "passage": str(row["passage"]), "label": label})

    if not examples:
        raise ValueError("Dataset must contain query, passage, and binary label fields.")
    return examples


def prepare_dataset(data_path: str) -> Dataset:
    """Load validated JSON or JSONL query/passage relevance examples."""
    return Dataset.from_list(_load_rows(data_path))


def train_reranker(data_path: str, output_dir: str):
    dataset = prepare_dataset(data_path)
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Keep the base model's single-logit head: runtime reranking uses a scalar score.
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)

    def tokenize_function(examples):
        return tokenizer(examples["query"], examples["passage"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["query", "passage"])
    tokenized = tokenized.rename_column("label", "labels")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA-tuned reranker to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder reranker with PEFT/LoRA")
    parser.add_argument("--data-path", type=str, required=True, help="Path to JSON or JSONL training data")
    parser.add_argument("--output-dir", type=str, default="artifacts/reranker", help="Output directory")
    args = parser.parse_args()
    train_reranker(args.data_path, args.output_dir)
