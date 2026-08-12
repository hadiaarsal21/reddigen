"""
Train the RELEVANCE RANKER.

Task: given (query, post) pairs, learn an embedding space where a query
about "SEO for SaaS" is close to posts about SEO for SaaS and far from
unrelated posts.

Architecture: Sentence-BERT (siamese/dual-encoder), trained with a
MultipleNegativesRankingLoss (in-batch negatives — no need for hand-
picked negatives, which are expensive to label).

Data: ml/data/relevance_pairs.jsonl with rows:
    {"query": "...", "positive": "post title\\n\\npost body"}

Run:
    python ml/train_relevance.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pairs(path: Path):
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(InputExample(texts=[row["query"], row["positive"]]))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/relevance_pairs.jsonl")
    ap.add_argument("--out", default="ml/models/relevance")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training data not found at {data_path}")

    examples = load_pairs(data_path)
    print(f"[train_relevance] Loaded {len(examples)} query-post pairs")

    model = SentenceTransformer(BASE_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_relevance] Training on {device}")

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=int(len(loader) * 0.1),
        show_progress_bar=True,
    )

    model.save(args.out)
    print(f"[train_relevance] Saved to {args.out}")


if __name__ == "__main__":
    main()
