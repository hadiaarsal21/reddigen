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
- [Feature ↔ model matrix](#feature--model-matrix)
- [End-to-end pipeline — Search](#end-to-end-pipeline--search)
- [End-to-end pipeline — Deep Scan](#end-to-end-pipeline--deep-scan)
- [End-to-end pipeline — Discover Subreddits](#end-to-end-pipeline--discover-subreddits)
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
| 5 | Reply generation | FLAN-T5-base + LoRA | 250M (0.4% trainable) | ~5K (query, post, reply) triples | `POST /generate-reply` |

All models are:

- **Trained from open pre-trained backbones** using PyTorch + HuggingFace
  Transformers — no proprietary APIs
- **Served from a single FastAPI process** (`ml/server.py`) so the Next.js
  app can chain them with plain HTTP calls
- **Independently swappable** — each has its own training script, its own
  checkpoint directory, and its own endpoint

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

- **Backbone:** DistilBERT-base-uncased (66M params, 12 layers, 768-dim).
  Chosen over full BERT for a 2× speed-up with < 3% loss in accuracy on
  comparable tasks.
- **Head:** single dense layer (768 → 4) with softmax
- **Loss:** cross-entropy
- **Max sequence length:** 256 tokens

### Training data

- **Format:** JSONL — `{"text": "title\n\nbody", "label": "buying_intent"}`
- **File:** `ml/data/intent_labeled.jsonl`
- **Volume:** the shipped sample is 48 examples for smoke testing.
  Real training needs **~10K rows** for a stable macro-F1 above 0.85.

### Running

```bash
python ml/train_intent.py                           # defaults
python ml/train_intent.py --epochs 5 --batch-size 32   # override
```

Checkpoint saves to `ml/models/intent/`. On next request, `server.py`
detects it via `_try_load_intent()` and switches from stub to real
inference.

### Integration

- **Used by:** Search
- **Called from:** `src/app/api/search/route.ts` for every raw Reddit post
- **Return contract:** `{"label": "...", "confidence": 0.0-1.0}`
- **Downstream:** posts where `label ∈ {buying_intent, advice_seeking}`
  proceed to the relevance ranker; others are dropped.

---

## Model 2 — Relevance Ranker

### Problem

Given a user query ("SEO agency for SaaS startups") and a piece of text
(a post, a comment, or a subreddit description), score how topically
relevant the text is on a `[0, 1]` scale.

### Architecture

- **Backbone:** Sentence-BERT (`all-MiniLM-L6-v2`, 22M params)
- **Design:** dual encoder / siamese network. Query and document are
  encoded independently into 384-dim vectors; relevance is cosine
  similarity.
- **Loss:** `MultipleNegativesRankingLoss` — for a batch of positive
  (query, post) pairs, every OTHER post in the batch is treated as an
  implicit negative.

### Why a bi-encoder, not a cross-encoder?

Cross-encoders are more accurate but must re-encode the (query, post)
pair for every candidate. A bi-encoder lets us pre-encode texts once and
score against arbitrary queries at inference cost of one query encode +
N dot products. For a real-time pipeline hitting dozens of items per
query, this is worth the small accuracy trade.

### Training data

- **Format:** `{"query": "...", "positive": "title\n\nbody"}`
- **File:** `ml/data/relevance_pairs.jsonl`
- **Volume:** ~15K pairs is a good target. Larger batches → more
  negatives → better loss signal.

### Running

```bash
python ml/train_relevance.py
```

### Integration

- **Used by:** Search (post relevance), Discover Subreddits (subreddit
  description relevance), Deep Scan (buyer-comment relevance to product)
- **Called from:** `src/app/api/search/route.ts`,
  `src/app/api/discover/route.ts`, `src/app/api/deep-scan/route.ts`
- **Return contract:** `{"score": 0.0-1.0}` (raw cosine similarity)

---

## Model 3 — Role Classifier (Buyer / Seller / Advisor)

### Problem

The **Deep Scan** model. When looking at COMMENTS under a Reddit post —
say, someone posted "[For Hire] SEO expert" and 40 people replied — we
need to distinguish:

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

- **Backbone:** RoBERTa-base (125M params)
- **Head:** dense (768 → 4) + softmax
- **Loss:** custom **focal loss** (Lin et al., 2017) with γ=2.0 and
  per-class weights inversely proportional to class frequency

### Why focal loss over plain cross-entropy?

Focal loss down-weights easy examples so the model focuses on hard
minority-class examples (buyers). Combined with inverse-frequency class
weights, this dramatically improves the buyer F1 (the metric we
actually care about) at a small cost to overall accuracy.

### Training data

- **Format:** `{"text": "comment body", "label": "buyer" | "seller" | "advisor" | "other"}`
- **File:** `ml/data/role_labeled.jsonl`
- **Volume:** ~8K comments minimum, ideally 15K with at least 1,500
  buyer examples.

### Running

```bash
python ml/train_role.py
```

### Integration

- **Used by:** Deep Scan (primary), Search (annotates final leads)
- **Called from:** `src/app/api/deep-scan/route.ts`,
  `src/app/api/search/route.ts`
- **Return contract:** `{"role": "buyer|seller|advisor|other", "confidence": 0.0-1.0}`

---

## Model 4 — Sentiment + Urgency (Multi-Task)

### Problem

For each lead, predict two things:

- **Sentiment**: positive / neutral / negative
- **Urgency**: low / medium / high

Both signals feed into priority ranking on the dashboard — a frustrated,
high-urgency lead is worth much more than a happy, casual one.

### Architecture — the interesting part

Instead of two separate models we use **hard parameter sharing**: one
shared RoBERTa encoder + two independent linear heads.

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
  encoder produces both predictions
- **Feature sharing** — both tasks depend on emotional intensity in the
  text, so a shared representation empirically outperforms two
  independent models by 1-3 F1 points on each task
- **Implicit regularisation** — reduces overfitting on the smaller of
  the two datasets

### Loss

Weighted sum of two cross-entropy losses:

```
L_total = α · L_sentiment + (2 - α) · L_urgency
```

### Training data

- **Format:** `{"text": "...", "sentiment": "...", "urgency": "..."}`
- **File:** `ml/data/sentiment_labeled.jsonl`
- **Volume:** 12K rows with balanced sentiment classes and slightly
  over-represented `high` urgency

### Running

```bash
python ml/train_sentiment.py
```

### Integration

- **Used by:** Search, Deep Scan
- **Return contract:** `{"sentiment": "...", "urgency": "..."}`

---

## Model 5 — Reply Generator

### Problem

Given `(query, post_title, post_body, tone)`, generate a natural
Reddit-style reply the user could send in one click.

### Architecture

- **Backbone:** FLAN-T5-base (250M params, encoder-decoder)
- **Fine-tuning method:** **LoRA** (Hu et al., 2021) — Low-Rank
  Adaptation. Instead of updating all 250M parameters, we train small
  rank-8 update matrices attached to the attention Q and V projections.

### Why LoRA?

- Trainable params drop from 250M to ~1M (0.4%)
- Fits on a 12-16 GB consumer GPU without gradient checkpointing
- Training is ~4× faster than full fine-tuning
- No accuracy loss on our task
- Weight merging at the end means inference doesn't need `peft`

### Prompt template

```
Write a natural, helpful Reddit reply.
tone: {tone}
our offer: {query}
post title: {title}
post body: {body}
reply:
```

The tone token conditions the model — same input with `tone: professional`
vs `tone: casual` produces measurably different reply styles.

### Training data

- **Format:** `{"query": "...", "post_title": "...", "post_body": "...", "tone": "helpful", "reply": "the gold reply"}`
- **File:** `ml/data/reply_pairs.jsonl`
- **Volume:** ~5K rows minimum. Human-authored gold replies are
  expensive but critical.

### Running

```bash
python ml/train_reply.py
```

### Integration

- **Used by:** Search, Deep Scan
- **Return contract:** `{"reply": "..."}`

---

## Serving architecture

`ml/server.py` is a single FastAPI process that:

1. Lazy-loads each model's checkpoint the first time its endpoint is hit
2. Falls back to a rule-based stub if the checkpoint isn't present
3. Uses HuggingFace `pipeline()` for all inference
4. Auto-detects CUDA and moves models to GPU if available

Model boundaries are enforced by the HTTP interface — the Next.js app has
no direct dependency on PyTorch, HuggingFace, or any ML library. Swapping
backbones (e.g., DistilBERT → DeBERTa) is a pure Python change that the
app never notices.

### Health endpoint

`GET /` returns which checkpoints are currently loaded. The DashboardShell
sidebar polls this and shows a live green/red status dot.

---

## Feature ↔ model matrix

| Feature | Models used | Where |
|---|---|---|
| **Search** | Intent (1), Relevance (2), Sentiment+Urgency (4), Role (3), Reply (5) | `src/app/api/search/route.ts` |
| **Deep Scan** | Role (3) primary, Relevance (2), Sentiment+Urgency (4), Reply (5) | `src/app/api/deep-scan/route.ts` |
| **Discover Subreddits** | Relevance (2) — used in doc-retrieval mode | `src/app/api/discover/route.ts` |
| **Leads** | none (pure storage) | `src/app/api/leads/route.ts` |

---

## End-to-end pipeline — Search

```
user query
    │
    ▼
searchReddit(query)                → N raw posts (~30-50)
    │
    ▼
[Model 1] classifyIntent(post)     → keep only buying_intent + advice_seeking (drops ~90%)
    │
    ▼
[Model 2] scoreRelevance(post)     → cosine(query, post); drop below 0.20
    │
    ▼
[Model 4] predictSentiment(post)   → tag each with sentiment + urgency (multi-task)
    │
    ▼
[Model 3] classifyRole(post)       → attribute role (buyer/seller/advisor)
    │
    ▼
[Model 5] generateReply(post)      → tone-conditioned draft reply
    │
    ▼
scored, ranked, reply-drafted leads
```

**Typical latency on GPU:** 3–6 seconds for a 30-post query.
**Typical latency on CPU:** 15–30 seconds.
**With stubs (no models loaded):** < 1 second.

---

## End-to-end pipeline — Deep Scan

Deep Scan is the "comment mining" feature. Standard search only reads post
titles + bodies — but the highest-intent buyers often appear as REPLIES
under other people's posts. Someone posts "[For Hire] SEO expert" and 40
people reply. Most of those replies are sellers or advisors. The few
buyers hidden in that thread are the leads Deep Scan surfaces.

```
user product description ("SEO agency for SaaS")
    │
    ▼
Build buyer-attracting queries:
  "[for hire] SEO agency for SaaS"
  "hiring SEO agency for SaaS"
  "recommend SEO agency for SaaS"
  "looking for SEO agency for SaaS"
  "SEO agency for SaaS freelancer"
    │
    ▼
searchReddit(each query)              → dedupe by post ID, rank by comment count
    │
    ▼
Pick top ~10 posts with the most comments (more comments = more chance of
buyers replying)
    │
    ▼
fetchComments(each post)              → up to 25 comments per post
    │
    ▼
[Model 3] classifyRole(comment)       → for EVERY comment: buyer/seller/advisor/other
    │
    ▼
Keep only role=="buyer" & confidence >= 0.55
    │
    ▼
For each surviving buyer comment:
  [Model 2] scoreRelevance(product, comment)    → how relevant is this comment to your specific product?
  [Model 4] predictSentiment(comment)           → sentiment + urgency of the buyer
  [Model 5] generateReply(product, parent_post, comment, tone) → draft a reply that references the thread
    │
    ▼
Rank by relevance; display as commented-lead cards
```

This is where the role classifier's focal loss + class weights matter
most — without them the model just predicts "seller" for everything
and we'd catch zero buyers.

---

## End-to-end pipeline — Discover Subreddits

Discover uses the **relevance ranker in document-retrieval mode** — the
first (and only) feature to use it that way. Instead of scoring one
query against one post, we score one query against the descriptions of
every candidate subreddit and rank.

```
user product description ("cold email deliverability tool")
    │
    ▼
searchReddit(product, sort=new)   ─┐
                                    ├→ union, dedup → ~100 posts
searchReddit(product, sort=rel)   ─┘
    │
    ▼
Group posts by subreddit; count mentions per subreddit
    │
    ▼
For each of the top ~25 subreddits by mention count:
  fetchSubredditInfo(name)              → title, public_description, subscribers, active_users
    │
    ▼
[Model 2] scoreRelevance(product_query, subreddit_description)
                                        → cosine similarity in the same shared 384-dim space
    │
    ▼
composite_score = 0.7 * semantic + 0.3 * log(1 + mentions) / log(51)
    │
    ▼
Rank by composite; return with subscriber counts + sample matching posts
```

Why the composite score? Pure semantic ranking surfaces obscure
communities that happen to mention the topic verbatim. Pure activity
ranking surfaces huge general subreddits. The 70/30 blend prefers
communities that are both semantically aligned AND actively discussing
the topic — the ones you'd actually want to post in.

---

## Evaluation methodology

Every training script prints train/val macro-F1 at each epoch. For a
proper writeup, extend with these evaluations:

### For the classification models (intent, role, sentiment, urgency)

- **Confusion matrix** — `sklearn.metrics.confusion_matrix`
- **Per-class precision/recall/F1** — already printed by `train_role.py`;
  adapt for the others
- **Held-out subreddit test** — split by subreddit, not by row. Reserve
  5–10 subreddits your model has never seen — this measures true
  generalisation, not just fit.

### For the relevance ranker

- **Recall@K** — for a held-out set of (query, positive_post) pairs,
  what fraction of positives are ranked in the top K candidates?
- **Mean Reciprocal Rank (MRR)** — reciprocal of the rank of the first
  relevant result, averaged across queries
- **For subreddit-retrieval mode:** curate a gold set of (product, best
  subreddit) pairs and report recall@5 of the discover pipeline

### For the reply generator

- **BLEU / ROUGE-L** against the gold reply — captures surface-level
  overlap, but poorly correlated with quality on open-ended generation
- **Human preference study** — for 50-100 queries, generate a reply
  and ask 3 evaluators to rank the model reply vs. the gold reply vs.
  a rule-based baseline
- **Toxicity + safety** — run outputs through a Detoxify classifier

### For the role classifier (Deep Scan specifically)

- **Buyer-precision at retrieval time** — of the comments Deep Scan
  surfaces as buyers, what fraction actually ARE buyers on human
  audit? Target ≥ 0.85 for a demo-worthy pipeline.
- **Buyer-recall** — of ALL true buyers in a fresh test thread, what
  fraction did Deep Scan surface? Target ≥ 0.70.

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

**For the reply task specifically:**
Budget realistically 2-4 minutes per label (writing a natural gold
reply). Consider bootstrapping with a stronger model to generate silver
labels, then editing rather than authoring from scratch.

---

*End of guide. For code-level docs, see the docstrings at the top of
each training script and `ml/server.py`.*
