from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


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


def prepare_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load validated JSON or JSONL query/passage relevance examples."""
    return _load_rows(data_path)


def train_reranker(data_path: str, output_dir: str):
    dataset = prepare_dataset(data_path)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, revision=RERANKER_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        revision=RERANKER_REVISION,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    batch_size = 8

    for start in range(0, len(dataset), batch_size):
        rows = dataset[start : start + batch_size]
        features = tokenizer(
            [row["query"] for row in rows],
            [row["passage"] for row in rows],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**features).logits
        if logits.shape[-1] == 1:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels.long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned reranker to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder reranker with a bounded PyTorch loop")
    parser.add_argument("--data-path", type=str, required=True, help="Path to JSON or JSONL training data")
    parser.add_argument("--output-dir", type=str, default="artifacts/reranker", help="Output directory")
    args = parser.parse_args()
    train_reranker(args.data_path, args.output_dir)
