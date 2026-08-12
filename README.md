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
reddigen-local/
├── src/                    # Next.js 15 web app (dashboard, leads, models)
│   ├── app/                # Pages + API routes
│   ├── components/         # Nav
│   └── lib/                # DB, Reddit fetch, ML client
├── ml/                     # Python ML layer
│   ├── server.py           # FastAPI inference server (all 5 models)
│   ├── train_intent.py     # DistilBERT — intent classifier
│   ├── train_relevance.py  # Sentence-BERT — relevance ranker
│   ├── train_role.py       # RoBERTa — buyer/seller/advisor
│   ├── train_sentiment.py  # RoBERTa multi-task — sentiment + urgency
│   ├── train_reply.py      # FLAN-T5 + LoRA — reply generator
│   ├── data/               # Sample labelled JSONL for each task
│   ├── models/             # Trained checkpoints go here (gitignored)
│   └── requirements.txt
├── prisma/
│   └── schema.prisma       # SQLite schema for the Leads table
├── MODELS-GUIDE.md         # Detailed ML architecture + training guide
└── README.md               # (You are here)
```

---

## Prerequisites

- **Node.js 20+** and npm
- **Python 3.10+** and pip
- **Optional:** CUDA-capable GPU for training the transformer models (inference
  works fine on CPU, just slower)

---

## Quick start (5 minutes to a working demo)

Open two terminals.

### Terminal 1 — Set up and run the ML server

```bash
# From the project root
cd reddigen-local

# Install Python deps (creates the FastAPI server + all training deps)
pip install -r ml/requirements.txt

# Start the model server on http://localhost:8000
python ml/server.py
```

On first boot, the server will report which trained checkpoints are loaded.
If you haven't trained any yet, every endpoint uses a rule-based **stub**
that returns plausible outputs — enough for the whole app to work
end-to-end. Once you drop trained checkpoints into `ml/models/<name>/`, the
server automatically switches to real inference on the next request.

### Terminal 2 — Set up and run the Next.js app

```bash
# From the same project root
cp .env.example .env
npm install
npx prisma generate
npx prisma db push        # creates the SQLite file at prisma/dev.db

npm run dev               # http://localhost:3000
```

Open <http://localhost:3000>, go to **Dashboard**, type any query (e.g.
"python freelancer for automation") and press **Run search**. The pipeline
retrieves Reddit posts, runs them through all five models, and shows scored
leads with drafted replies.

---

## Training the models

Sample labelled data ships in `ml/data/*.jsonl` so all five training scripts
run out of the box. For real evaluation-quality models, expand each file to
5,000–10,000 rows (see `ml/data/README.md` for schema and public data
sources).

```bash
# Train each model (independently — order doesn't matter)
python ml/train_intent.py       # ~2-5 min on GPU, ~30 min on CPU
python ml/train_relevance.py    # ~2-5 min
python ml/train_role.py         # ~5-10 min
python ml/train_sentiment.py    # ~5-10 min
python ml/train_reply.py        # ~20-40 min (largest model)
```

Each script saves to `ml/models/<name>/`. The FastAPI server picks up the
new checkpoints **without a restart** — the next request to that endpoint
switches from stub to real inference.

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
| `npm run dev` | Start Next.js on `:3000` |
| `npm run build` | Production build |
| `npm run ml:setup` | Install Python deps |
| `npm run ml:serve` | Start FastAPI on `:8000` |
| `npm run db:push` | Sync SQLite schema |
| `npm run db:studio` | Open Prisma Studio (visual DB browser) |
| `python ml/train_<task>.py` | Train the model for `<task>` |

---

## License

MIT — see LICENSE.
