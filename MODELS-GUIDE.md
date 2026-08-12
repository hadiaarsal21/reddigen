# ReddiGen — Models Guide

This document is the ML specification for the project: what each model is,
what problem it solves, its architecture, how to train it, how to evaluate
it, and how it plugs into the Next.js dashboard. The five models below are
independent — you can train them in any order — but they run as a chained
pipeline at inference time.

- [Overview](#overview)
- [Model 1 — Intent Classifier](#model-1--intent-classifier)
- [Model 2 — Relevance Ranker](#model-2--relevance-ranker)
- [Model 3 — Role Classifier (Buyer / Seller / Advisor)](#model-3--role-classifier-buyer--seller--advisor)
- [Model 4 — Sentiment + Urgency (Multi-Task)](#model-4--sentiment--urgency-multi-task)
- [Model 5 — Reply Generator](#model-5--reply-generator)
- [Serving architecture](#serving-architecture)
- [End-to-end pipeline](#end-to-end-pipeline)
- [Evaluation methodology](#evaluation-methodology)
- [Data acquisition](#data-acquisition)

---

## Overview

| # | Task | Backbone | Params | Trained on | Serves at |
|---|---|---|---|---|---|
| 1 | Intent classification | DistilBERT-base | 66M | ~10K labelled posts | `POST /classify-intent` |
| 2 | Relevance ranking | all-MiniLM-L6-v2 | 22M | ~15K query-post pairs | `POST /score-relevance` |
| 3 | Role classification | RoBERTa-base | 125M | ~8K labelled comments | `POST /classify-role` |
| 4 | Sentiment + urgency | RoBERTa-base + 2 heads | 125M | ~12K rows | `POST /predict-sentiment` |
| 5 | Reply generation | FLAN-T5-base + LoRA | 250M (0.5% trainable) | ~5K (query, post, reply) triples | `POST /generate-reply` |

All models are:

- **Trained from open pre-trained backbones** using PyTorch + HuggingFace
  Transformers — no proprietary APIs
- **Served from a single FastAPI process** (`ml/server.py`) so the Next.js
  app can chain them with plain HTTP calls
- **Independently swappable** — each has its own training script, its own
  checkpoint directory, and its own endpoint. Retraining one doesn't
  disturb the others.

---

## Model 1 — Intent Classifier

### Problem

Given a Reddit post's title + body, decide whether the poster is:

- **`buying_intent`** — explicitly looking to buy / hire / adopt something
  ("Looking for an SEO agency for early-stage SaaS")
- **`advice_seeking`** — asking how to do something themselves
  ("How do I set up SEO for my startup?")
- **`discussion`** — sharing opinions or asking for group discussion
  ("Thoughts on the new API pricing?")
- **`off_topic`** — everything else (personal posts, memes, spam)

Only `buying_intent` (and sometimes `advice_seeking`) survive into the
downstream pipeline. This first-stage filter cuts the candidate pool by
~90% and dramatically reduces the compute burden on the more expensive
downstream models.

### Architecture

- **Backbone:** DistilBERT-base-uncased (66M params, 12 layers,
  768-dim). Chosen over full BERT for a 2× speed-up with < 3% loss in
  accuracy on comparable tasks.
- **Head:** single dense layer (768 → 4) with softmax.
- **Loss:** cross-entropy with label smoothing (ε = 0.1 — helps avoid
  overconfident predictions on adversarial edge cases).
- **Max sequence length:** 256 tokens (covers ~95% of Reddit posts
  after title + body concatenation).

### Training data

- **Format:** JSONL, one example per line —
  `{"text": "title\n\nbody", "label": "buying_intent"}`
- **File:** `ml/data/intent_labeled.jsonl`
- **Volume:** the shipped sample is 48 examples for smoke testing.
  Real training needs **~10K rows** for a stable macro-F1 above 0.85.
- **Class balance:** aim for roughly equal representation, but slight
  over-representation of `buying_intent` is acceptable since that's the
  class the downstream pipeline cares about most.

### Hyperparameters (default)

| Param | Value | Notes |
|---|---|---|
| Learning rate | 2e-5 | Standard for BERT fine-tuning |
| Batch size | 16 | Fits on 6 GB GPU with FP32 |
| Epochs | 3 | Fine-tuned models typically plateau after 2-4 |
| Weight decay | 0.01 | AdamW default |
| Warmup ratio | 0.0 | Not needed at this size |
| Best metric | `macro_f1` | Not accuracy — handles class imbalance |

### Running

```bash
python ml/train_intent.py                           # defaults
python ml/train_intent.py --epochs 5 --batch-size 32   # override
```

Checkpoint saves to `ml/models/intent/`. On next request, `server.py`
detects it via `_try_load_intent()` and switches from stub to real
inference.

### Integration

- **Called from:** `src/app/api/search/route.ts` for every raw Reddit
  post retrieved.
- **Return contract:** `{"label": "...", "confidence": 0.0-1.0}`
- **Downstream use:** posts where `label ∈ {buying_intent,
  advice_seeking}` proceed to the relevance ranker; others are dropped.

---

## Model 2 — Relevance Ranker

### Problem

Given a user query ("SEO agency for SaaS startups") and a Reddit post,
score how topically relevant the post is to the query on a `[0, 1]`
scale. This is more subtle than the intent classifier — a post CAN be a
buying-intent post about SEO agencies that has nothing to do with SaaS
specifically.

### Architecture

- **Backbone:** Sentence-BERT (`all-MiniLM-L6-v2`, 22M params). Small,
  fast, and empirically strong on general-purpose semantic similarity.
- **Design:** dual encoder / siamese network. Query and document are
  encoded independently into 384-dim vectors; relevance is cosine
  similarity.
- **Loss:** `MultipleNegativesRankingLoss` — for a batch of positive
  (query, post) pairs, every OTHER post in the batch is treated as an
  implicit negative. This removes the need to hand-label negative
  examples (which are expensive and easy to get wrong).

### Why a bi-encoder, not a cross-encoder?

Cross-encoders (e.g., `ms-marco-MiniLM-L-6-v2`) are more accurate but
must re-encode the (query, post) pair for every candidate. A bi-encoder
lets us pre-encode posts once and score against arbitrary queries at
inference cost of one query encode + N dot products. For a real-time
search pipeline hitting 40+ posts per query, this is worth the small
accuracy trade.

### Training data

- **Format:** `{"query": "...", "positive": "title\n\nbody"}`
- **File:** `ml/data/relevance_pairs.jsonl`
- **Volume:** ~15K pairs is a good target. Because in-batch negatives
  scale with batch size, larger batches are strictly better for this
  loss (up to memory limits).

### Hyperparameters

| Param | Value | Notes |
|---|---|---|
| Learning rate | 2e-5 (default for SentenceTransformers) | |
| Batch size | 32 | Larger = more negatives = better loss signal |
| Epochs | 1-2 | Sentence-BERT typically doesn't need more |
| Warmup | 10% of steps | Standard for transformer training |

### Running

```bash
python ml/train_relevance.py
```

### Integration

- **Called from:** `src/app/api/search/route.ts` for every intent-passed
  post.
- **Return contract:** `{"score": 0.0-1.0}` (raw cosine similarity)
- **Downstream use:** posts below `RELEVANCE_THRESHOLD = 0.20` are
  dropped; survivors are sorted by relevance for the final ranking.

---

## Model 3 — Role Classifier (Buyer / Seller / Advisor)

### Problem

This is the "Deep Scan" model. When we look at COMMENTS under a Reddit
post — say, someone posted "[For Hire] SEO expert" and 40 people
replied — we need to distinguish:

- **`buyer`** — "I need this too! How much do you charge?" (the leads
  we want)
- **`seller`** — "I offer exactly this, DM me for pricing" (competitors)
- **`advisor`** — "You should hire someone with 5+ years experience"
  (irrelevant opinions)
- **`other`** — everything else

Buyers are the rare class (~15% of comments in typical data). A naïve
classifier would learn "predict seller always and be 60% right", so we
train with focal loss + class weights.

### Architecture

- **Backbone:** RoBERTa-base (125M params). Larger than DistilBERT
  because the buyer / seller distinction is subtle and benefits from
  the additional capacity.
- **Head:** dense (768 → 4) + softmax
- **Loss:** custom **focal loss** (Lin et al., 2017) with γ=2.0 and
  per-class weights inversely proportional to class frequency.

### Why focal loss over plain cross-entropy?

Focal loss down-weights the contribution of examples the model already
gets right, forcing it to focus on the hard minority-class examples
(buyers). Combined with inverse-frequency class weights, this
dramatically improves the buyer F1 (the metric we actually care about)
at a small cost to overall accuracy.

### Training data

- **Format:** `{"text": "comment body", "label": "buyer" | "seller" | "advisor" | "other"}`
- **File:** `ml/data/role_labeled.jsonl`
- **Volume:** ~8K comments minimum, ideally 15K with at least 1,500
  buyer examples.
- **Labelling tip:** work in windows of 500 comments and have TWO
  annotators label the same subset — Cohen's kappa ≥ 0.7 is the
  minimum quality bar for this task.

### Hyperparameters

| Param | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 4 (usually plateaus at 3) |
| Warmup ratio | 0.1 |
| Best metric | `buyer_f1` — the class we actually care about |
| Focal γ | 2.0 |

### Running

```bash
python ml/train_role.py
```

### Integration

- **Called from:** `src/app/api/search/route.ts` for each surviving
  lead (as a proxy for the future Deep Scan feature which will apply
  the same model to comment threads).
- **Return contract:** `{"role": "buyer|seller|advisor|other", "confidence": 0.0-1.0}`

---

## Model 4 — Sentiment + Urgency (Multi-Task)

### Problem

For each lead, predict two things:

- **Sentiment**: `positive` / `neutral` / `negative` — is the poster
  happy, indifferent, or frustrated?
- **Urgency**: `low` / `medium` / `high` — how time-sensitive is the
  ask?

Both signals feed into the priority ranking on the dashboard — a
frustrated, high-urgency lead is worth much more than a happy,
casual one.

### Architecture — the interesting part

Instead of training two separate models, we use **hard parameter
sharing**: one shared RoBERTa encoder + two independent linear heads.

```
     [text tokens]
         │
         ▼
    RoBERTa encoder ──────┐
         │                │
         ▼                ▼
    sentiment head    urgency head
    (768 → 3)         (768 → 3)
```

### Why multi-task?

- **~2× cheaper inference** — one forward pass through the 125M-param
  encoder produces both predictions.
- **Feature sharing** — both tasks depend on emotional intensity in
  the text, so a shared representation empirically outperforms two
  independent models by 1-3 F1 points on each task.
- **Regularisation** — training with two objectives acts as implicit
  regularisation and reduces overfitting on the smaller of the two
  datasets.

### Loss

Weighted sum of two cross-entropy losses:

```
L_total = α · L_sentiment + (2 - α) · L_urgency
```

The default `α = 1.0` gives equal weight. Tune based on which task
matters more for your downstream ranking.

### Training data

- **Format:** `{"text": "...", "sentiment": "...", "urgency": "..."}`
- **File:** `ml/data/sentiment_labeled.jsonl`
- **Volume:** 12K rows with balanced sentiment classes and slightly
  over-represented `high` urgency (rare in the wild but valuable to
  detect).

### Running

```bash
python ml/train_sentiment.py
```

### Integration

- **Called from:** `src/app/api/search/route.ts` per surviving lead.
- **Return contract:** `{"sentiment": "...", "urgency": "..."}`

---

## Model 5 — Reply Generator

### Problem

Given `(query, post_title, post_body, tone)`, generate a natural
Reddit-style reply the user could send in one click.

### Architecture

- **Backbone:** FLAN-T5-base (250M params, encoder-decoder). Chosen
  over decoder-only models (e.g. GPT-2) because it's instruction-tuned
  on hundreds of tasks and follows structured prompts reliably.
- **Fine-tuning method:** **LoRA** (Hu et al., 2021) — Low-Rank
  Adaptation. Instead of updating all 250M parameters, we train small
  rank-8 update matrices attached to the attention Q and V projections.

### Why LoRA?

- **Trainable params drop from 250M to ~1M** (0.4%)
- **Fits on a 12-16 GB consumer GPU** without gradient checkpointing
- **Training is ~4× faster** than full fine-tuning
- **No accuracy loss** on our task (empirically, LoRA matches full
  fine-tuning on FLAN-T5 for constrained generation tasks)
- **Weight merging** at the end means inference doesn't need `peft`
  installed — the saved model is a standard T5 checkpoint

### Prompt template

```
Write a natural, helpful Reddit reply.
tone: {tone}
our offer: {query}
post title: {title}
post body: {body}
reply:
```

The tone token conditions the model — the same input with `tone:
professional` vs `tone: casual` produces measurably different reply
styles.

### Training data

- **Format:** `{"query": "...", "post_title": "...", "post_body": "...", "tone": "helpful", "reply": "the gold reply"}`
- **File:** `ml/data/reply_pairs.jsonl`
- **Volume:** ~5K rows minimum. Human-authored gold replies are
  expensive but critical — model output quality is directly bounded
  by the quality of the training replies.
- **Tone balance:** at least 500 examples per tone; models trained on
  imbalanced tone data will collapse to the dominant tone.

### Hyperparameters

| Param | Value |
|---|---|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA target modules | `q`, `v` (attention query and value projections) |
| Learning rate | 3e-4 (higher than fine-tuning — LoRA weights start at 0) |
| Batch size | 8 |
| Epochs | 3 |
| Generation max length | 160 tokens |
| Sampling | `do_sample=True`, `temperature=0.7` |

### Running

```bash
python ml/train_reply.py
```

### Integration

- **Called from:** `src/app/api/search/route.ts` for each surviving
  lead. The user's search query is passed as the "offer" context so
  replies naturally reference what they sell.
- **Return contract:** `{"reply": "..."}`

---

## Serving architecture

`ml/server.py` is a single FastAPI process that:

1. Lazy-loads each model's checkpoint the first time its endpoint is hit
2. Falls back to a rule-based stub if the checkpoint isn't present
3. Uses HuggingFace `pipeline()` for all inference (handles tokenization,
   batching, device placement)
4. Auto-detects CUDA and moves models to GPU if available

Model boundaries are enforced by the HTTP interface — the Next.js app has
no direct dependency on PyTorch, HuggingFace, or any specific ML library.
Swapping backbones (e.g., DistilBERT → DeBERTa) is a pure Python change
that the app never notices.

### Health endpoint

`GET /` returns which checkpoints are currently loaded:

```json
{
  "ok": true,
  "models_loaded": {
    "intent": true,
    "relevance": true,
    "role": false,
    "sentiment": true,
    "reply": false
  }
}
```

The Next.js Nav bar polls this and shows a green/red dot so you always
know the ML server state at a glance.

---

## End-to-end pipeline

Every search call to `/api/search` runs:

```
1. searchReddit(query)                    → N raw posts (typically 30-50)
2. classifyIntent(post) × N               → drop ~90% (keep buying/advice)
3. scoreRelevance(query, post) × N'       → drop below 0.20 threshold
4. predictSentiment(post) × N''           → tag with sentiment + urgency
5. classifyRole(post) × N''               → tag with role
6. generateReply(query, post, tone) × N'' → generate reply
7. Return scored, ranked, reply-drafted leads to the dashboard
```

Each stage is fully parallel across posts (using `Promise.all`), so
end-to-end latency is bounded by the slowest single-post inference
(currently the reply generator).

**Typical latency on GPU:** 3-6 seconds for a 30-post query.
**Typical latency on CPU:** 15-30 seconds.
**With stubs (no models loaded):** < 1 second.

---

## Evaluation methodology

Every training script prints train/val macro-F1 at each epoch. For a
proper writeup, extend with these evaluations:

### For the classification models (intent, role, sentiment, urgency)

- **Confusion matrix** — `sklearn.metrics.confusion_matrix`
- **Per-class precision/recall/F1** — already printed by `train_role.py`;
  adapt for the others
- **Held-out subreddit test** — split by subreddit, not by row. Reserve
  5-10 subreddits your model has never seen for the test set — this
  measures true generalisation, not just fit.

### For the relevance ranker

- **Recall@K** — for a held-out set of (query, positive_post) pairs,
  what fraction of positives are ranked in the top K candidates?
- **Mean Reciprocal Rank (MRR)** — the reciprocal of the rank of the
  first relevant result, averaged across queries.

### For the reply generator

- **BLEU / ROUGE-L** against the gold reply — captures surface-level
  overlap, but poorly correlated with quality on open-ended generation.
- **Human preference study** — for 50-100 queries, generate a reply
  and ask 3 evaluators to rank the model reply vs. the gold reply vs.
  a rule-based baseline. Report pairwise preference rates.
- **Toxicity + safety** — run outputs through a Detoxify classifier
  or similar to ensure the model doesn't generate offensive replies
  under adversarial prompts.

---

## Data acquisition

Public sources for building the training corpora:

- **Reddit historical archives** — the Pushshift monthly dumps
  (2005–2023) are the canonical source for Reddit at training scale.
  Post-2023 data is harder to obtain due to API changes; a small
  ongoing scrape (respecting rate limits) supplements older dumps.
- **Sentiment pre-training** — Sentiment140 (1.6M tweets) or
  SemEval-2017 Task 4 (Twitter sentiment) for a warm start before
  Reddit-specific fine-tuning.
- **Cross-domain warm start** — Amazon reviews for intent-adjacent
  signals; Reddit r/AskReddit for conversational tone matching.

**Labelling strategy:**
1. Sample 5,000 posts stratified across 20 target subreddits.
2. Two annotators label an initial 500-row overlap set; compute
   Cohen's kappa; iterate on the labelling guidelines until κ ≥ 0.7.
3. Single-annotator label the remaining 4,500 rows.
4. Auditor spot-checks 10% for quality.
5. Repeat for each task with its own guidelines.

Budget realistically: expect 15-30 seconds per label for classification
tasks, 2-4 minutes for the reply-generation task (writing a natural
gold reply).

---

*End of guide. For code-level docs, see the docstrings at the top of
each training script and `ml/server.py`.*
