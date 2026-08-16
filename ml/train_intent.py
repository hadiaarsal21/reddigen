"""
Train the INTENT CLASSIFIER.

Task: given a Reddit post (title + body), classify it into one of:
  - buying_intent     ("looking for a python expert")
  - advice_seeking    ("how do I set up SEO for my startup")
  - discussion        ("thoughts on the new API pricing?")
  - off_topic         (everything else)

Architecture: fine-tuned DistilBERT (small, fast, GPU-optional).

Data: expects a JSONL file at ml/data/intent_labeled.jsonl with rows like
    {"text": "post title\\n\\npost body", "label": "buying_intent"}

Run:
    python ml/train_intent.py

The trained model is saved to ml/models/intent/ and the ML server picks it
up automatically on next request — no server restart needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlflow_utils  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "distilbert-base-uncased"
LABELS = ["buying_intent", "advice_seeking", "discussion", "off_topic"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"macro_f1": macro_f1, "precision": p, "recall": r, "f1": f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/intent_labeled.jsonl")
    ap.add_argument("--out", default="ml/models/intent")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Training data not found at {data_path}. See ml/data/README.md for the "
            "expected schema and how to generate a starter labelled set."
        )

    rows = load_jsonl(data_path)
    print(f"[train_intent] Loaded {len(rows)} labelled examples")

    mlflow_utils.init("reddigen-intent")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = Dataset.from_list([
        {"text": r["text"], "label": LABEL2ID[r["label"]]} for r in rows if r.get("label") in LABEL2ID
    ])
    ds = ds.train_test_split(test_size=0.15, seed=42)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_intent] Training on {device}")

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=mlflow_utils.report_to(),
    )

    params = {
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_len": args.max_len,
        "num_labels": len(LABELS),
        "device": device,
    }

    with mlflow_utils.run(f"intent-{BASE_MODEL}", params):
        mlflow_utils.log_dataset(rows, label_key="label")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            processing_class=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"[train_intent] Final metrics: {metrics}")

        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"[train_intent] Saved to {args.out}")

        if mlflow_utils.MLFLOW_AVAILABLE:
            import mlflow

            mlflow.log_metrics({
                f"final.{k.replace('eval_', '')}": v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            })
        mlflow_utils.log_checkpoint(args.out)


if __name__ == "__main__":
    main()
