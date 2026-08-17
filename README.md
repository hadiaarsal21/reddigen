# ReddiGen — Local ML Research Build

An end-to-end deep-learning system that discovers, classifies, and ranks
buying-intent conversations on Reddit. Five custom-trained models power the
pipeline. Everything runs locally — no external API keys required.

> **This is the academic/research build.** The production SaaS version lives
> at [`reddit-lead-gen`](https://github.com/projectsbyfarhan1107/reddit-lead-gen).
> This build strips out billing, auth, and third-party services so it can be
> reviewed, evaluated, and demonstrated entirely on `localhost`.

---

## What's inside

```
reddigen/
├── src/                    # Next.js 15 web app (dashboard, leads, models)
│   ├── app/                # Pages + API routes
│   ├── components/         # Nav, search panel, chip filters
│   └── lib/                # DB, Reddit fetch, ML client, request limits
├── ml/                     # Python ML layer
│   ├── server.py           # FastAPI inference server (all 5 models)
│   ├── mlflow_utils.py     # Experiment tracking shared by every script
│   ├── import_models.py    # Unpack trained checkpoints from Kaggle
│   ├── train_intent.py     # DistilBERT — intent classifier
│   ├── train_relevance.py  # Sentence-BERT — relevance ranker
│   ├── train_role.py       # RoBERTa — buyer/seller/advisor
│   ├── train_sentiment.py  # RoBERTa multi-task — sentiment + urgency
│   ├── train_reply.py      # FLAN-T5 + LoRA — reply generator
│   ├── data/               # Labelled JSONL + generate_dataset.py
│   ├── models/             # Trained checkpoints go here (gitignored)
│   └── requirements.txt
├── notebooks/
│   └── reddigen_kaggle_training.ipynb   # GPU training on Kaggle
├── prisma/
│   └── schema.prisma       # SQLite schema for the Leads table
├── MODELS-GUIDE.md         # ML architecture — what each model is and why
├── TRAINING.md             # How to build the data, train, and import models
└── README.md               # (You are here)
```

---

## Prerequisites

- **Node.js 20+** and npm
- **Python 3.10+** and pip
- **Optional:** CUDA-capable GPU for training the transformer models (inference
  works fine on CPU, just slower)

---

## Quick start

### One-time setup

```bash
# Node side
npm install
cp .env.example .env          # Windows: copy .env.example .env
npx prisma generate
npx prisma db push            # creates the SQLite database

# Python side
python -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.txt
# Windows: .venv\Scripts\python.exe -m pip install -r ml/requirements.txt
```

### Run it

```bash
npm start
```

That is the whole thing. One command starts both the FastAPI ML server on
`:8000` and the Next.js app on `:3000`, waits until each is actually
answering, and shuts both down together on Ctrl+C.

```
ReddiGen (production)

[ml]  starting FastAPI on port 8000
[ml]  ready on http://localhost:8000
[web] starting Next.js on port 3000
[web]  ✓ Ready in 1150ms

  ReddiGen is running
  http://localhost:3000
```

Use `npm run dev` for the same thing with hot reload.

The launcher checks its prerequisites first, so a missing virtualenv or an
occupied port produces one clear message rather than a half-started stack. It
creates `.env` from the example if absent, and runs `next build` on the first
production start.

Open <http://localhost:3000>, go to **Search**, type what you sell (e.g.
"python freelancer for automation") and press **Run Search**. The pipeline
retrieves Reddit posts, runs them through all five models, and shows scored
leads with drafted replies you can copy straight into the thread.

### Running without trained models

With `ml/models/` empty, every endpoint falls back to a rule-based **stub**
that returns plausible output, so the whole app works end to end before any
training has happened. Drop trained checkpoints into `ml/models/<name>/` and
the server switches to real inference on the next request, no restart needed.
See [TRAINING.md](TRAINING.md).

---

## Training the models

Build the datasets, then train:

```bash
python ml/data/generate_dataset.py   # 50,000 rows across the five tasks

python ml/train_intent.py       # ~2-5 min on GPU, ~30 min on CPU
python ml/train_relevance.py    # ~2-5 min
python ml/train_role.py         # ~5-10 min
python ml/train_sentiment.py    # ~5-10 min
python ml/train_reply.py        # ~20-40 min (largest model)
```

Each script saves to `ml/models/<name>/`. The FastAPI server picks up the
new checkpoints **without a restart** — the next request to that endpoint
switches from stub to real inference.

**No GPU?** The two RoBERTa models and the FLAN-T5 reply generator are not
practical on CPU. `notebooks/reddigen_kaggle_training.ipynb` runs the whole
pipeline on a free Kaggle T4 in about an hour, then
`python ml/import_models.py reddigen-models.zip` brings the checkpoints back.

**Experiment tracking.** Every run logs hyperparameters, dataset balance,
hardware, per-epoch metrics and final evaluation to MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Note the shipped datasets are **synthetic** — generated from templates rather
than human-labelled — so held-out scores validate the pipeline rather than
measuring generalisation to live Reddit. [`TRAINING.md`](TRAINING.md) covers
this in full, along with the Kaggle setup and troubleshooting.

See [`MODELS-GUIDE.md`](MODELS-GUIDE.md) for architecture details,
hyperparameters, evaluation methodology, and how each model integrates with
the Next.js app.

---

## Architecture

```
┌────────────────────────┐        ┌────────────────────────┐
│   Next.js  :3000       │        │  FastAPI  :8000        │
│                        │        │                        │
│   /dashboard  ────┐    │  HTTP  │   /classify-intent     │
│   /leads          │    │───────▶│   /score-relevance     │
│   /models         │    │        │   /classify-role       │
│   /api/search   ──┘    │        │   /predict-sentiment   │
│   /api/leads           │        │   /generate-reply      │
│                        │        │                        │
│   Prisma → SQLite      │        │   torch + HuggingFace  │
└────────────────────────┘        └────────────────────────┘
              │
              ▼
     reddit.com (public JSON)
```

Every trained model is served from the FastAPI process. The Next.js API
routes chain the models for each request. SQLite stores saved leads. Reddit
is queried directly using its public search JSON.

---

## What's a stub vs. real?

- **Real:** the Next.js app, the FastAPI server, all 5 training scripts, the
  full pipeline chaining, SQLite storage, direct Reddit fetching, sample
  labelled data.
- **Stub (when no checkpoint present):** the model endpoints return
  rule-based outputs that match the trained-model schema exactly. This lets
  you demo the app immediately, and lets a grader verify the training
  pipeline works even if they don't want to sit through GPU training.

Once real checkpoints exist in `ml/models/<name>/`, the endpoints
transparently switch. No code changes.

---

## Model summary

| Task | Backbone | Loss | Sample training data |
|---|---|---|---|
| Intent classification | DistilBERT | Cross-entropy | `intent_labeled.jsonl` |
| Relevance ranking | Sentence-BERT (MiniLM) | Multiple negatives ranking | `relevance_pairs.jsonl` |
| Role (Buyer/Seller/Advisor) | RoBERTa-base | Focal loss + class weights | `role_labeled.jsonl` |
| Sentiment + urgency (multi-task) | RoBERTa-base, shared encoder | Joint cross-entropy | `sentiment_labeled.jsonl` |
| Reply generation | FLAN-T5-base + LoRA | Seq2seq | `reply_pairs.jsonl` |

Full detail: [`MODELS-GUIDE.md`](MODELS-GUIDE.md).

---

## Commands reference

| Command | Purpose |
|---|---|
| **`npm start`** | **Start everything: ML server + web app** |
| `npm run dev` | Same, with hot reload |
| `npm run build` | Production build only |
| `npm run web` | Web app alone, no ML server |
| `npm run ml:serve` | ML server alone |
| `npm run ml:setup` | Install Python deps |
| `npm run db:push` | Sync SQLite schema |
| `npm run db:studio` | Open Prisma Studio (visual DB browser) |
| `python ml/data/generate_dataset.py` | Build the training corpora |
| `python ml/train_<task>.py` | Train the model for `<task>` |
| `python ml/import_models.py <zip>` | Install checkpoints trained on Kaggle/Colab |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Browse tracked experiments |

Ports are configurable: `PORT` for the web app, `ML_PORT` for the ML server.

---

## License

MIT — see LICENSE.
