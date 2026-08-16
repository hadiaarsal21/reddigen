"""
Train the MULTI-TASK SENTIMENT + URGENCY MODEL.

Instead of training two separate models we share a single RoBERTa
encoder and stick two classification heads on top — one for sentiment
(positive/neutral/negative), one for urgency (low/medium/high). This
gives us:
  * ~2x cheaper inference (one encode, two predictions)
  * a shared representation, which is empirically stronger on both
    tasks when they share underlying features (both depend on emotional
    intensity in the text)

Data: ml/data/sentiment_labeled.jsonl with rows:
    {"text": "...", "sentiment": "positive"|"neutral"|"negative",
     "urgency": "low"|"medium"|"high"}

Run:
    python ml/train_sentiment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlflow_utils  # noqa: E402
from sklearn.metrics import f1_score
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "roberta-base"

SENTIMENT_LABELS = ["positive", "neutral", "negative"]
URGENCY_LABELS = ["low", "medium", "high"]
S2I = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
U2I = {l: i for i, l in enumerate(URGENCY_LABELS)}


class MultiTaskModel(PreTrainedModel):
    """Shared RoBERTa encoder + two independent classification heads.
    The forward pass returns a combined loss that weights both tasks
    equally by default (adjust `sentiment_weight` if one matters more)."""

    def __init__(self, config, sentiment_weight: float = 1.0):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.sentiment_head = nn.Linear(hidden, len(SENTIMENT_LABELS))
        self.urgency_head = nn.Linear(hidden, len(URGENCY_LABELS))
        self.sentiment_weight = sentiment_weight
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        sentiment_labels=None,
        urgency_labels=None,
        **kwargs,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        s_logits = self.sentiment_head(pooled)
        u_logits = self.urgency_head(pooled)

        loss = None
        if sentiment_labels is not None and urgency_labels is not None:
            fn = nn.CrossEntropyLoss()
            s_loss = fn(s_logits, sentiment_labels)
            u_loss = fn(u_logits, urgency_labels)
            loss = self.sentiment_weight * s_loss + (2 - self.sentiment_weight) * u_loss

        return {"loss": loss, "sentiment_logits": s_logits, "urgency_logits": u_logits}


class MTDataCollator:
    """Pads text batches and stacks the two label tensors."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        s_labels = torch.tensor([f.pop("sentiment_labels") for f in features])
        u_labels = torch.tensor([f.pop("urgency_labels") for f in features])
        batch = self._pad(features)
        batch["sentiment_labels"] = s_labels
        batch["urgency_labels"] = u_labels
        return batch


class MTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs["loss"].detach() if outputs.get("loss") is not None else None
        # Concatenate the two logits so metrics can split them apart again
        logits = torch.cat([outputs["sentiment_logits"], outputs["urgency_logits"]], dim=1)
        labels = torch.stack([inputs["sentiment_labels"], inputs["urgency_labels"]], dim=1)
        return loss, logits, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    n_s = len(SENTIMENT_LABELS)
    s_preds = np.argmax(logits[:, :n_s], axis=-1)
    u_preds = np.argmax(logits[:, n_s:], axis=-1)
    s_true, u_true = labels[:, 0], labels[:, 1]
    return {
        "sentiment_f1": f1_score(s_true, s_preds, average="macro"),
        "urgency_f1": f1_score(u_true, u_preds, average="macro"),
        "avg_f1": (
            f1_score(s_true, s_preds, average="macro")
            + f1_score(u_true, u_preds, average="macro")
        ) / 2,
    }


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/sentiment_labeled.jsonl")
    ap.add_argument("--out", default="ml/models/sentiment")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training data not found at {data_path}")

    rows = load_jsonl(data_path)
    print(f"[train_sentiment] Loaded {len(rows)} rows")

    mlflow_utils.init("reddigen-sentiment-urgency")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = Dataset.from_list([
        {
            "text": r["text"],
            "sentiment_labels": S2I[r["sentiment"]],
            "urgency_labels": U2I[r["urgency"]],
        }
        for r in rows if r.get("sentiment") in S2I and r.get("urgency") in U2I
    ])
    ds = ds.train_test_split(test_size=0.15, seed=42)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    model = MultiTaskModel(cfg)
    # Load pretrained weights into the encoder
    pretrained = AutoModel.from_pretrained(BASE_MODEL)
    model.encoder.load_state_dict(pretrained.state_dict())

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="avg_f1",
        greater_is_better=True,
        logging_steps=10,
        report_to=mlflow_utils.report_to(),
        label_names=["sentiment_labels", "urgency_labels"],
    )

    params = {
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_len": args.max_len,
        "architecture": "shared encoder + 2 classification heads",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    with mlflow_utils.run(f"sentiment-urgency-{BASE_MODEL}", params):
        mlflow_utils.log_dataset(rows, label_key="sentiment")
        if mlflow_utils.MLFLOW_AVAILABLE:
            import mlflow

            from collections import Counter

            for lbl, n in sorted(Counter(r["urgency"] for r in rows if "urgency" in r).items()):
                mlflow.log_param(f"dataset.urgency.{lbl}", n)

        trainer = MTTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            processing_class=tok,
            data_collator=MTDataCollator(tok),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"[train_sentiment] Final metrics: {metrics}")

        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"[train_sentiment] Saved to {args.out}")

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
