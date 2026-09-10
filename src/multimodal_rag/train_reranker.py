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


def train_reranker(
    data_path: str,
    output_dir: str,
    *,
    model_name: str = RERANKER_MODEL,
    revision: str | None = RERANKER_REVISION,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
):
    """Fine-tune a cross-encoder and save a self-contained local checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if revision is None and not Path(model_name).is_dir():
        raise ValueError("Remote base models require a pinned revision; omit it only for a local checkpoint directory.")
    dataset = prepare_dataset(data_path)
    labels = [row["label"] for row in dataset]
    if len(set(labels)) < 2:
        raise ValueError("Training data must include both relevant and non-relevant examples.")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    generator = torch.Generator().manual_seed(42)
    for _epoch in range(epochs):
        order = torch.randperm(len(dataset), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            rows = [dataset[index] for index in order[start : start + batch_size]]
            features = tokenizer(
                [row["query"] for row in rows],
                [row["passage"] for row in rows],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            batch_labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**features).logits
            if logits.shape[-1] == 1:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], batch_labels)
            else:
                loss = torch.nn.functional.cross_entropy(logits, batch_labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    manifest = {
        "base_model": model_name,
        "base_revision": revision,
        "examples": len(dataset),
        "positive_examples": int(sum(labels)),
        "negative_examples": int(len(labels) - sum(labels)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": 42,
    }
    (output_path / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fine-tuned reranker to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a cross-encoder reranker with a bounded PyTorch loop")
    parser.add_argument("--data-path", type=str, required=True, help="Path to JSON or JSONL training data")
    parser.add_argument("--output-dir", type=str, default="artifacts/reranker", help="Output directory")
    parser.add_argument("--model", type=str, default=RERANKER_MODEL, help="Base Hugging Face cross-encoder model or local checkpoint")
    parser.add_argument("--revision", type=str, default=RERANKER_REVISION, help="Pinned base-model revision; use an empty string for a local checkpoint")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    train_reranker(
        args.data_path,
        args.output_dir,
        model_name=args.model,
        revision=args.revision.strip() or None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
