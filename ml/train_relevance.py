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
import sys
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlflow_utils  # noqa: E402

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pairs(path: Path):
    examples = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            row = json.loads(line)
            examples.append(InputExample(texts=[row["query"], row["positive"]]))
    return examples


def evaluate_retrieval(model, pairs, ks=(1, 5, 10)) -> dict:
    """
    Rank every held-out positive against the whole held-out pool.

    For each query the correct post should rank first. Reports Recall@K and
    MRR — the metrics MODELS-GUIDE.md specifies for the bi-encoder, and far
    more informative than the training loss alone.
    """
    if len(pairs) < 2:
        return {}
    queries = [q for q, _ in pairs]
    docs = [d for _, d in pairs]

    q_emb = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    d_emb = model.encode(docs, convert_to_tensor=True, normalize_embeddings=True)
    sims = q_emb @ d_emb.T                      # (Q, D) cosine similarity
    ranks = sims.argsort(dim=1, descending=True)

    gold = torch.arange(len(pairs), device=ranks.device).unsqueeze(1)
    positions = (ranks == gold).float().argmax(dim=1)  # 0-indexed rank of gold

    out = {"mrr": float((1.0 / (positions + 1)).mean())}
    for k in ks:
        if k <= len(pairs):
            out[f"recall_at_{k}"] = float((positions < k).float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/relevance_pairs.jsonl")
    ap.add_argument("--out", default="ml/models/relevance")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.1)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training data not found at {data_path}")

    examples = load_pairs(data_path)
    print(f"[train_relevance] Loaded {len(examples)} query-post pairs")

    # Held-out split for retrieval metrics
    import random

    rng = random.Random(42)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(2, int(len(shuffled) * args.val_split))
    val_examples = shuffled[:n_val]
    train_examples = shuffled[n_val:]
    val_pairs = [(e.texts[0], e.texts[1]) for e in val_examples]
    print(f"[train_relevance] train={len(train_examples)} val={len(val_pairs)}")

    mlflow_utils.init("reddigen-relevance")

    model = SentenceTransformer(BASE_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_relevance] Training on {device}")

    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    params = {
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": "MultipleNegativesRankingLoss",
        "val_split": args.val_split,
        "device": device,
    }

    with mlflow_utils.run(f"relevance-{BASE_MODEL.split('/')[-1]}", params):
        mlflow_utils.log_dataset([{"pair": 1} for _ in examples])

        before = evaluate_retrieval(model, val_pairs)
        print(f"[train_relevance] Before fine-tuning: {before}")

        model.fit(
            train_objectives=[(loader, loss)],
            epochs=args.epochs,
            warmup_steps=int(len(loader) * 0.1),
            show_progress_bar=True,
        )

        after = evaluate_retrieval(model, val_pairs)
        print(f"[train_relevance] After fine-tuning:  {after}")

        model.save(args.out)
        print(f"[train_relevance] Saved to {args.out}")

        if mlflow_utils.MLFLOW_AVAILABLE:
            import mlflow

            mlflow.log_metrics({f"baseline.{k}": v for k, v in before.items()})
            mlflow.log_metrics({f"final.{k}": v for k, v in after.items()})
        mlflow_utils.log_checkpoint(args.out)


if __name__ == "__main__":
    main()
