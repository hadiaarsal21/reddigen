"""
Train the ROLE CLASSIFIER (BUYER / SELLER / ADVISOR).

This is the heart of Deep Scan — when we look at comments UNDER Reddit
posts (e.g. under a "[For Hire] SEO expert" thread), we need to filter
out the sellers pitching their services and the advisors dispensing
opinions, and keep only the actual BUYERS saying "I need this too".

Architecture: fine-tuned RoBERTa-base with focal loss to handle the
built-in class imbalance (buyers are the rare class, sellers dominate).

Data: ml/data/role_labeled.jsonl with rows:
    {"text": "comment body", "label": "buyer" | "seller" | "advisor" | "other"}

Run:
    python ml/train_role.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlflow_utils  # noqa: E402

from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "roberta-base"
LABELS = ["buyer", "seller", "advisor", "other"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017). Down-weights easy examples so the
    model focuses on the minority BUYER class instead of just learning
    'predict seller all the time and be right 60% of the time'."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class WeightedTrainer(Trainer):
    """Trainer subclass that swaps in our focal loss."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = FocalLoss(alpha=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.criterion(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


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
    buyer_f1 = f1_score(labels, preds, labels=[LABEL2ID["buyer"]], average="macro")
    return {"macro_f1": macro_f1, "buyer_f1": buyer_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/role_labeled.jsonl")
    ap.add_argument("--out", default="ml/models/role")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training data not found at {data_path}")

    rows = load_jsonl(data_path)
    print(f"[train_role] Loaded {len(rows)} labelled comments")

    mlflow_utils.init("reddigen-role")

    # Compute class weights inversely proportional to frequency so the
    # focal loss can further down-weight the majority class.
    counts = {l: 0 for l in LABELS}
    for r in rows:
        if r.get("label") in counts:
            counts[r["label"]] += 1
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(LABELS) * counts[l]) if counts[l] > 0 else 1.0 for l in LABELS],
        dtype=torch.float,
    )
    print(f"[train_role] Class counts: {counts}")
    print(f"[train_role] Class weights: {weights.tolist()}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = Dataset.from_list([
        {"text": r["text"], "label": LABEL2ID[r["label"]]}
        for r in rows if r.get("label") in LABEL2ID
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
    weights = weights.to(device)

    # 10% warmup expressed in steps: transformers 5.x dropped warmup_ratio,
    # while warmup_steps exists in both 4.x and 5.x.
    steps_per_epoch = max(1, math.ceil(len(ds["train"]) / args.batch_size))
    warmup_steps = int(0.1 * steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="buyer_f1",  # optimise for the rare class
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
        "loss": "focal(gamma=2.0)+inverse_freq_weights",
        "device": device,
    }

    with mlflow_utils.run(f"role-{BASE_MODEL}", params):
        mlflow_utils.log_dataset(rows, label_key="label")

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            processing_class=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
            class_weights=weights,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"[train_role] Final metrics: {metrics}")

        # Detailed per-class report
        preds = np.argmax(trainer.predict(ds["test"]).predictions, axis=-1)
        labels = np.array([ex["label"] for ex in ds["test"]])
        report = classification_report(labels, preds, target_names=LABELS)
        print("\n" + report)

        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"[train_role] Saved to {args.out}")

        if mlflow_utils.MLFLOW_AVAILABLE:
            import mlflow

            mlflow.log_metrics({
                f"final.{k.replace('eval_', '')}": v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            })
            # per-class precision/recall/F1 for the audit trail
            detail = classification_report(
                labels, preds, target_names=LABELS, output_dict=True, zero_division=0
            )
            for cls in LABELS:
                if cls in detail:
                    for m in ("precision", "recall", "f1-score"):
                        mlflow.log_metric(f"{cls}.{m.replace('-score', '')}", detail[cls][m])
            mlflow.log_text(report, "classification_report.txt")
        mlflow_utils.log_checkpoint(args.out)


if __name__ == "__main__":
    main()
