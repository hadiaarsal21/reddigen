"""
Fine-tune the REPLY GENERATOR.

Task: given (query, post_title, post_body, tone), generate a natural
Reddit reply the user could send in one click.

Architecture: FLAN-T5-base fine-tuned with LoRA (parameter-efficient —
only ~0.5% of the parameters are updated, making this trainable on a
single consumer GPU in a few hours).

Data: ml/data/reply_pairs.jsonl with rows:
    {
        "query":       "SEO agency for SaaS startups",
        "post_title":  "Anyone know a good SEO agency for early-stage SaaS?",
        "post_body":   "I'm looking for someone who...",
        "tone":        "helpful",
        "reply":       "the human-written gold-standard reply"
    }

Run:
    python ml/train_reply.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlflow_utils  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_MODEL = "google/flan-t5-base"

PROMPT = (
    "Write a natural, helpful Reddit reply.\n"
    "tone: {tone}\n"
    "our offer: {query}\n"
    "post title: {title}\n"
    "post body: {body}\n"
    "reply:"
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/reply_pairs.jsonl")
    ap.add_argument("--out", default="ml/models/reply")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-input", type=int, default=512)
    ap.add_argument("--max-output", type=int, default=160)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Training data not found at {data_path}")

    rows = load_jsonl(data_path)
    print(f"[train_reply] Loaded {len(rows)} (query, post, reply) triples")

    mlflow_utils.init("reddigen-reply")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # LoRA config — targets the attention projection matrices only. Rank 8
    # is a sweet spot: enough capacity to specialise, small enough to train
    # on 12-16 GB of VRAM without gradient checkpointing.
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q", "v"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    def format_row(row):
        prompt = PROMPT.format(
            tone=row.get("tone", "helpful"),
            query=row["query"],
            title=row["post_title"],
            body=(row.get("post_body") or "")[:1200],
        )
        return {"input_text": prompt, "target_text": row["reply"]}

    ds = Dataset.from_list([format_row(r) for r in rows])
    ds = ds.train_test_split(test_size=0.1, seed=42)

    def tokenize(batch):
        model_inputs = tok(
            batch["input_text"],
            truncation=True,
            max_length=args.max_input,
        )
        labels = tok(
            batch["target_text"],
            truncation=True,
            max_length=args.max_output,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_reply] Training on {device}")

    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        generation_max_length=args.max_output,
        logging_steps=10,
        report_to=mlflow_utils.report_to(),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    params = {
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "peft": "LoRA",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": "q,v",
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / max(1, total), 4),
        "device": device,
    }

    with mlflow_utils.run(f"reply-{BASE_MODEL.split('/')[-1]}-lora", params):
        mlflow_utils.log_dataset(rows, label_key="tone")

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            processing_class=tok,
            data_collator=DataCollatorForSeq2Seq(tok, model=model),
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"[train_reply] Final metrics: {metrics}")

        # Merge LoRA weights back into the base model so `server.py` can load
        # it with a plain `pipeline(...)` call without needing peft at inference
        # time (simpler deploy). Comment out to keep LoRA-only weights instead.
        merged = model.merge_and_unload()
        merged.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"[train_reply] Saved (merged) model to {args.out}")

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
