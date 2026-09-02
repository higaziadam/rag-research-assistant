from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def prepare_dataset(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    examples = []
    for row in rows:
        if "query" in row and "passage" in row and "label" in row:
            examples.append(
                {
                    "query": row["query"],
                    "passage": row["passage"],
                    "label": int(row["label"]),
                }
            )

    if not examples:
        raise ValueError("Dataset must contain query, passage, and label fields.")

    ds = load_dataset("json", data_files={"train": data_path})
    return ds["train"], examples


def train_reranker(data_path: str, output_dir: str):
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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
        return tokenizer(
            list(zip(examples["query"], examples["passage"])),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    dataset = load_dataset("json", data_files={"train": data_path})
    tokenized = dataset["train"].map(tokenize_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=20,
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved LoRA-tuned reranker to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder reranker with PEFT/LoRA")
    parser.add_argument("--data-path", type=str, required=True, help="Path to JSONL or JSON dataset")
    parser.add_argument("--output-dir", type=str, default="artifacts/reranker", help="Output folder")
    args = parser.parse_args()
    train_reranker(args.data_path, args.output_dir)
