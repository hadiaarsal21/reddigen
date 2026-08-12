# ReddiGen — Full Project Context

> A single reference document for anyone reading this repo — reviewers,
> collaborators, or you six months from now. Covers what the project is,
> why every design decision was made, how the pieces fit together, how
> to run it, and where the trade-offs live. Read this once and you have
> the whole picture.

## Table of contents

- [1. What ReddiGen is](#1-what-reddigen-is)
- [2. The problem it solves](#2-the-problem-it-solves)
- [3. What's in this repository (vs. the production sibling)](#3-whats-in-this-repository-vs-the-production-sibling)
- [4. Architecture at a glance](#4-architecture-at-a-glance)
- [5. The ML layer](#5-the-ml-layer)
- [6. The web layer](#6-the-web-layer)
- [7. How Reddit data is fetched](#7-how-reddit-data-is-fetched)
- [8. Data storage](#8-data-storage)
- [9. Feature deep-dive — Search](#9-feature-deep-dive--search)
- [10. Feature deep-dive — Deep Scan](#10-feature-deep-dive--deep-scan)
- [11. Feature deep-dive — Discover Subreddits](#11-feature-deep-dive--discover-subreddits)
- [12. Design system + UI theme](#12-design-system--ui-theme)
- [13. Setup + run](#13-setup--run)
- [14. Training the real models](#14-training-the-real-models)
- [15. Development workflow](#15-development-workflow)
- [16. What's intentionally NOT here + why](#16-whats-intentionally-not-here--why)
- [17. Where to look for what](#17-where-to-look-for-what)
- [18. Known trade-offs and future work](#18-known-trade-offs-and-future-work)

---

## 1. What ReddiGen is

ReddiGen is an end-to-end deep-learning system that discovers, classifies,
and ranks buying-intent conversations on Reddit — turning the world's
largest open discussion forum into a real-time source of qualified sales
leads.

The system chains **five custom-trained transformer models**:

1. Intent classification — is the poster expressing buying intent?
2. Semantic relevance ranking — how well does this post match the user's product?
3. Role classification — buyer, seller, or advisor?
4. Multi-task sentiment + urgency — emotional tone and time-sensitivity
5. Tone-conditioned reply generation — draft a one-click response

All five models are trained from open pre-trained backbones (DistilBERT,
Sentence-BERT / MiniLM, RoBERTa, FLAN-T5) on labelled Reddit data. **No
proprietary APIs are used at any point** — inference happens on a local
FastAPI service that the Next.js dashboard calls over HTTP.

This is the **academic / research build** — the sibling production repo
[`reddit-lead-gen`](https://github.com/projectsbyfarhan1107/reddit-lead-gen)
adds billing, auth, cron monitoring, and multi-cloud proxy rotation for
the paid product. Everything in this repo runs on `localhost` with zero
external services.

---

## 2. The problem it solves

Reddit hosts hundreds of thousands of active communities where users
publicly ask for product recommendations, describe pain points, and
compare vendors. Every day, sentences like these get posted:

- *"Looking for an SEO agency for early-stage SaaS — anyone have recommendations?"*
- *"Tried 5 lead-gen tools, all garbage. What works in 2026?"*
- *"My ecommerce store traffic dropped 40% — need help fast."*

These are buyers raising their hand. For businesses, this is the highest-
intent, most under-tapped source of sales conversations on the internet.
The problem: those signals are **scattered across thousands of subreddits,
buried in comment threads, and expressed in natural language that keyword
search alone cannot capture**.

Manual triage doesn't scale — even a dedicated sales team can only skim
the top few subreddits and misses most of the volume. Keyword monitors
are either too noisy (thousands of false positives) or too narrow (miss
90% of relevant conversations). This is fundamentally an NLP problem.

ReddiGen solves it with five specialised models chained together:

- **Intent classifier** cuts the noise ~90% by keeping only posts that
  express buying intent.
- **Relevance ranker** semantically scores how well each surviving post
  matches the user's specific offer — catching relevant posts that don't
  share any exact keywords with the query.
- **Role classifier** in Deep Scan mines comment threads for the
  highest-intent lead source — buyers replying "I need this too" under
  someone else's post.
- **Sentiment + urgency** prioritises leads by emotional state and time-
  sensitivity.
- **Reply generator** turns each qualified lead into a ready-to-send
  reply drafted in the user's chosen tone.

---

## 3. What's in this repository (vs. the production sibling)

The production repo [`reddit-lead-gen`](https://github.com/projectsbyfarhan1107/reddit-lead-gen)
is a full SaaS with billing, auth, admin panel, monitors, 24/7 cron,
Cloudflare Workers proxy, Telegram notifications, and marketing site.
This repo — the **research / local build** — deliberately strips that
down to what matters for the ML story.

| Feature | Local build | Production build |
|---|---|---|
| Home | ✅ Simple overview | ✅ Marketing landing page |
| Dashboard (Search) | ✅ | ✅ |
| Leads | ✅ | ✅ |
| Deep Scan | ✅ | ✅ |
| Discover Subreddits | ✅ | ✅ |
| ML Models page | ✅ (bonus in-app architecture docs) | — |
| Auth | ❌ (single user) | ✅ Clerk |
| Billing | ❌ | ✅ Freemius |
| Monitors (24/7 cron) | ❌ | ✅ |
| Hidden Gems | ❌ | ✅ |
| Reddit fetch | ✅ Direct | ✅ Multi-cloud proxy rotation |
| DB | ✅ SQLite | ✅ Supabase PostgreSQL |
| ML inference | ✅ Local trained models via FastAPI | Uses external LLM API |
| Model training scripts | ✅ 5 complete scripts | — |
| Deployment | Local only | Hetzner VPS + Caddy + pm2 |

---

## 4. Architecture at a glance

```
┌────────────────────────────────────────────────────────────────────────┐
│                            Next.js  :3000                              │
│                                                                        │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐    │
│   │   /      │   │  /dashboard  │   │  /deep-scan  │   │ /leads   │    │
│   │ (home)   │   │  (search)    │   │  (comments)  │   │ (saved)  │    │
│   └────┬─────┘   └──────┬───────┘   └──────┬───────┘   └────┬─────┘    │
│        │                │                  │                │          │
│        │  ┌─────────────┴─────┐  ┌─────────┴──────┐  ┌──────┴──────┐   │
│        │  │/api/search        │  │/api/deep-scan  │  │/api/leads   │   │
│        │  │/api/discover      │  │                │  │             │   │
│        │  └────────┬──────────┘  └───────┬────────┘  └──────┬──────┘   │
│        │           │                     │                  │          │
│        │           │  HTTP JSON          │                  │          │
│        │           ▼                     ▼                  │          │
│        │  ┌────────────────────────────────────┐            │          │
│        │  │        FastAPI ML service :8000    │            │          │
│        │  │                                    │            │          │
│        │  │   POST /classify-intent            │            │          │
│        │  │   POST /score-relevance            │            │          │
│        │  │   POST /classify-role              │            │          │
│        │  │   POST /predict-sentiment          │            │          │
│        │  │   POST /generate-reply             │            │          │
│        │  │                                    │            │          │
│        │  │   PyTorch + HuggingFace + LoRA     │            │          │
│        │  │   auto-loads ml/models/<name>/     │            │          │
│        │  └────────────────────────────────────┘            │          │
│        │                                                    │          │
│        └────────────────┐                                   │          │
│                         ▼                                   ▼          │
│              www.reddit.com (public JSON)         SQLite (dev.db)      │
└────────────────────────────────────────────────────────────────────────┘
```

- **Next.js** is the frontend + API layer. Every user-facing page and
  every backend route lives here. TypeScript throughout.
- **FastAPI** is the ML inference layer. Python. Every trained model
  is served from this single process behind a REST interface. The
  Next.js side has zero dependency on PyTorch or HuggingFace — models
  are behind an HTTP boundary.
- **SQLite** is storage. One file. Prisma ORM. No database server needed.
- **Reddit** is queried directly against public JSON endpoints. For a
  localhost demo this is fine — no proxy needed at low request volume.

---

## 5. The ML layer

Full architectural details are in [`MODELS-GUIDE.md`](MODELS-GUIDE.md).
Quick summary here:

| # | Model | Backbone | Params | Loss | Endpoint |
|---|---|---|---|---|---|
| 1 | Intent classifier | DistilBERT-base | 66M | Cross-entropy | `POST /classify-intent` |
| 2 | Relevance ranker | Sentence-BERT (MiniLM) | 22M | MultipleNegativesRankingLoss | `POST /score-relevance` |
| 3 | Role classifier | RoBERTa-base | 125M | Focal loss + class weights | `POST /classify-role` |
| 4 | Sentiment + urgency | RoBERTa multi-task | 125M | Joint cross-entropy (2 heads) | `POST /predict-sentiment` |
| 5 | Reply generator | FLAN-T5-base + LoRA | 250M (0.4% trainable) | Seq2seq | `POST /generate-reply` |

Each model has:
- Its own training script at `ml/train_<task>.py` — complete, runnable
- Its own labelled JSONL data at `ml/data/<task>_labeled.jsonl` — sample
  files ship in-repo; scale per `ml/data/README.md`
- Its own endpoint on the shared FastAPI process
- A rule-based fallback stub the server uses when the checkpoint is
  absent — so the app always works, even before any training

**Stub → real transition is transparent.** The Next.js side has no idea
whether the ML server is returning stub outputs or real trained-model
predictions. When you train a model and save its checkpoint to
`ml/models/<name>/`, the FastAPI server picks it up automatically on
the next request. No restart, no code change.

---

## 6. The web layer

Next.js 15 (App Router) + React 19 + TypeScript.

### Pages

| Route | Purpose | Layout |
|---|---|---|
| `/` | Landing / overview | DashboardShell |
| `/dashboard` | Search UI — main feature | DashboardShell |
| `/leads` | Saved leads (SQLite) | DashboardShell |
| `/deep-scan` | Comment-mining feature | DashboardShell |
| `/discover` | Semantic subreddit ranking | DashboardShell |
| `/models` | In-app ML architecture docs | DashboardShell |

### API routes

| Route | Purpose |
|---|---|
| `POST /api/search` | Chains all 5 models for a query |
| `POST /api/deep-scan` | Mines comment threads for buyers |
| `POST /api/discover` | Semantic-ranks subreddits for a product |
| `GET / POST / PATCH / DELETE /api/leads` | SQLite CRUD for saved leads |
| `GET /api/ml-status` | Health check — proxies FastAPI's `/` endpoint |

### Shell + design

Every dashboard page is wrapped in `DashboardShell`, which provides:
- Left sidebar with grouped navigation (Workspace / Deep mining / Research)
- Top nav with breadcrumb
- Live ML-server status indicator in the sidebar footer
- Mobile hamburger drawer at ≤800px viewport
- Consistent design tokens (see [Design system](#12-design-system--ui-theme))

---

## 7. How Reddit data is fetched

Reddit publishes two public endpoints we consume:

- **Search JSON** —
  `https://www.reddit.com/search.json?q={query}&sort={sort}&t={time}&limit={n}`
- **Comments JSON** —
  `https://www.reddit.com/r/{subreddit}/comments/{postId}.json`
- **Subreddit about JSON** —
  `https://www.reddit.com/r/{subreddit}/about.json`

Wrapped in `src/lib/reddit.ts`. Each call:
- Uses a rotating user-agent string (real browser UAs)
- Retries on HTTP 429 with exponential backoff
- Times out at 15 seconds
- Returns typed `RedditPost`, `RedditComment`, or `SubredditInfo` objects

**No proxy needed** at localhost scale. The production sibling routes
these through a multi-cloud worker pool because at commercial-scale
traffic Reddit blocks the source IPs; at demo scale the direct fetch
just works.

---

## 8. Data storage

SQLite via Prisma. One file at `prisma/dev.db`. Zero setup.

Two tables:

### `leads`

Every saved lead — from Search, Deep Scan, or manual entry. Full schema
in `prisma/schema.prisma`. Key fields:
- `redditId` (unique) — the post ID or `t1_<commentId>` for Deep Scan
- `title`, `subreddit`, `url`, `author`, `selftext` — Reddit metadata
- `keyword` — the search query that surfaced this lead
- `relevanceScore`, `sentiment`, `urgency`, `leadType`, `role` — ML outputs
- `suggestedReply` — the generated reply draft
- `foundVia` — `search` | `deep_scan` | `discover` | `manual`
- `status` — `pending` | `replied` (user toggles from the Leads page)
- `parentPostId`, `parentPostUrl` — Deep Scan comments carry their parent
  post reference

### `discover_sessions`

Snapshot of a Discover Subreddits run. Currently written by the frontend
when the user saves a session; not used by the pipeline itself.

---

## 9. Feature deep-dive — Search

**Route:** `/dashboard` → `POST /api/search`
**Models used:** all 5

### Flow

1. User types what they sell (e.g. `"python freelancer for automation"`),
   picks a tone and time window.
2. Frontend `POST`s `{query, tone, time}` to `/api/search`.
3. Backend does two parallel Reddit searches (`sort=new` + `sort=relevance`)
   to broaden coverage. Dedupe by post ID. Cap at 40 raw posts.
4. For every raw post, in parallel:
   - Call `POST /classify-intent` on the FastAPI server (Model 1)
   - Call `POST /score-relevance` (Model 2)
5. Filter: keep only posts where intent ∈ {`buying_intent`, `advice_seeking`}
   AND relevance ≥ 0.20. Sort by relevance descending.
6. For every surviving post, in parallel:
   - Call `POST /predict-sentiment` (Model 4)
   - Call `POST /classify-role` (Model 3)
   - Call `POST /generate-reply` with the tone (Model 5)
7. Return the enriched list. Frontend renders it as scored lead cards
   with drafted replies.

### Typical numbers

- 30–50 raw posts from Reddit
- ~5–10 survive the intent + relevance filters
- ~5–8 make it into the final scored list
- Total wall time: 3–6 seconds on GPU, 15–30 seconds on CPU, < 1 second
  with stubs

---

## 10. Feature deep-dive — Deep Scan

**Route:** `/deep-scan` → `POST /api/deep-scan`
**Models used:** primarily Model 3 (role classifier); also 2, 4, 5

### Why it's the key differentiator

Standard search only looks at post titles + bodies. But the highest-
intent buyers frequently appear as REPLIES under other people's posts.
Someone posts `"[For Hire] SEO expert"` and 40 people reply — most are
sellers pitching their own services, a few are advisors dispensing
opinions, and hidden in there are the actual buyers saying `"I need this
too, how much do you charge?"`. Those are the Deep Scan finds.

### Flow

1. User types their product description.
2. Backend builds 5 buyer-attracting query variants:
   - `[for hire] {product}`
   - `hiring {product}`
   - `recommend {product}`
   - `looking for {product}`
   - `{product} freelancer`
3. Searches Reddit for each variant, dedupes by post ID.
4. Ranks posts by comment count — more comments means more chance of
   real buyers replying. Takes the top 10.
5. For each post, fetches up to 25 comments (top-level + threaded).
6. For EVERY comment, calls `POST /classify-role` (Model 3, the RoBERTa
   focal-loss classifier).
7. Keeps only role == `buyer` with confidence ≥ 0.55.
8. For each surviving buyer comment, runs Models 2, 4, 5 in parallel:
   - Relevance of the comment to the user's product
   - Sentiment + urgency of the buyer
   - Draft reply that references the parent post
9. Returns the ranked list along with statistics: posts scanned,
   comments examined, buyers found, sellers filtered, advisors filtered.

### Why the role classifier is the critical model here

Without focal loss + class weights, RoBERTa on this data would learn to
predict `"seller"` for everything and be 60% accurate — because sellers
dominate comment threads under service-offer posts. Focal loss
down-weights easy examples so the model focuses on hard buyer examples.
Inverse-frequency class weights further compensate. Combined, they lift
buyer-F1 from ~0.25 (naïve) to ~0.75 (properly weighted) on comparable
tasks.

---

## 11. Feature deep-dive — Discover Subreddits

**Route:** `/discover` → `POST /api/discover`
**Models used:** Model 2 (relevance ranker) — in document-retrieval mode

### Why this is interesting

Every other feature uses the relevance ranker to score **posts** against
a query. Discover uses it to score **subreddit descriptions** against a
query — the same model, different granularity. This is only possible
because the ranker was trained as a bi-encoder (dual encoder) — the same
embedding space works for any text length.

### Flow

1. User types their product description.
2. Backend does two broad Reddit searches (`sort=new` + `sort=relevance`),
   dedupes.
3. Groups posts by subreddit, counts mentions per subreddit.
4. Takes the top 25 subreddits by mention count.
5. For each candidate subreddit, fetches metadata (`about.json`) —
   description, subscriber count, active users.
6. For each subreddit with a description, calls `POST /score-relevance`
   comparing the user's product query to the subreddit's description.
7. Computes a composite score: **70% semantic similarity + 30% activity
   boost** (log-scaled mention count).
8. Returns the ranked list.

The composite score is deliberate — pure semantic ranking surfaces
obscure communities that happen to mention the topic; pure activity
ranking surfaces huge general subreddits. The 70/30 blend prefers
communities that are BOTH semantically aligned AND active on the topic —
the ones you'd actually want to post in.

---

## 12. Design system + UI theme

Mirrors the production ReddiGen dashboard design.

**Design tokens** (defined in `src/app/globals.css`):

```
Canvas:      #FAFAFA          Brand primary:  #FF4500 (orange)
Surface:     #FFFFFF          Brand hover:    #E63E00
Surface-alt: #F5F5F5          Brand tint:     #FFF4EF
Border:      #EBEBEB           Success:        #10B981
Text-primary:#18181B          Danger:         #EF4444
Text-secondary:#52525B         Purple:         #7C3AED
Text-tertiary:#8A8A93          Gold:           #F59E0B
```

**Typography**: Inter for UI, JetBrains Mono for code / meta metrics.

**Component conventions**:
- `.card` — white surface, subtle border, small shadow, 12px radius
- `.btn-primary` — orange gradient with drop shadow
- `.btn-ghost` — transparent with border, transitions to filled on hover
- `.pill` — small rounded badge with semantic variants (`hot`, `warm`,
  `cold`, `success`, `danger`, `purple`, `brand`, `neutral`)
- `.lead` — the tall lead card layout used across Search, Deep Scan, Leads
- `.reply-box` — orange-tinted italic quote block for generated replies
- `.sub-card` — the grid layout for Discover Subreddit results

The DashboardShell provides the full app chrome — sidebar with nav
groups, topbar with breadcrumb, live ML status indicator. Mobile drawer
kicks in at ≤ 800px.

---

## 13. Setup + run

### Prerequisites

- Node.js 20+ and npm
- Python 3.10+ and pip
- Optional: CUDA-capable GPU for training (inference works on CPU too)

### First-time setup

```bash
git clone https://github.com/projectsbyfarhan1107/reddigen-local.git
cd reddigen-local

# Node side
cp .env.example .env
npm install
npx prisma generate
npx prisma db push      # creates SQLite file at prisma/dev.db

# Python side
pip install -r ml/requirements.txt
```

### Run (two terminals)

**Terminal 1** — the ML server:
```bash
python ml/server.py     # http://localhost:8000
```

**Terminal 2** — the Next.js app:
```bash
npm run dev             # http://localhost:3000
```

Open http://localhost:3000, use the sidebar to try Search / Deep Scan /
Discover / Leads.

### Minimum-viable setup (skip torch install)

If you just want to see the UI + pipeline logic and don't need real
inference yet:

```bash
pip install fastapi uvicorn pydantic
python ml/server.py     # server runs on stubs only
```

The ML server will happily serve every endpoint with rule-based
stubs — the app works end-to-end, results are plausible-looking, and
nothing needs a GPU or a big model download.

---

## 14. Training the real models

Sample labelled data ships in `ml/data/*.jsonl` so all five scripts run
out of the box. For evaluation-quality models, scale each file to
5,000–10,000 rows per task following `ml/data/README.md`.

```bash
# Order doesn't matter — models are independent
python ml/train_intent.py       # DistilBERT
python ml/train_relevance.py    # Sentence-BERT
python ml/train_role.py         # RoBERTa + focal loss
python ml/train_sentiment.py    # RoBERTa multi-task
python ml/train_reply.py        # FLAN-T5 + LoRA (longest — 20-40 min GPU)
```

Each script saves to `ml/models/<name>/`. **The FastAPI server picks up
new checkpoints on the next request** — no restart, no code change. You
can literally train one model in one terminal while the app is running
in another and see the transition happen live.

---

## 15. Development workflow

### File layout

```
reddigen-local/
├── src/
│   ├── app/
│   │   ├── layout.tsx           # Root — imports globals.css + font
│   │   ├── globals.css          # Design system (all tokens + components)
│   │   ├── page.tsx             # Home (/)
│   │   ├── dashboard/page.tsx   # Search
│   │   ├── leads/page.tsx       # Saved leads
│   │   ├── deep-scan/page.tsx   # Comment mining
│   │   ├── discover/page.tsx    # Subreddit discovery
│   │   ├── models/page.tsx      # In-app ML docs
│   │   └── api/
│   │       ├── search/route.ts
│   │       ├── deep-scan/route.ts
│   │       ├── discover/route.ts
│   │       ├── leads/route.ts
│   │       └── ml-status/route.ts
│   ├── components/
│   │   ├── DashboardShell.tsx   # Sidebar + topbar shell
│   │   └── Icon.tsx             # Inline SVG icon set
│   └── lib/
│       ├── db.ts                # Prisma singleton
│       ├── reddit.ts            # Reddit fetch (posts/comments/subs)
│       └── mlClient.ts          # HTTP client for FastAPI ML server
├── prisma/schema.prisma         # SQLite schema
├── ml/
│   ├── server.py                # FastAPI — 5 endpoints, lazy-load, stubs
│   ├── train_intent.py          # DistilBERT training
│   ├── train_relevance.py       # Sentence-BERT training
│   ├── train_role.py            # RoBERTa + focal loss training
│   ├── train_sentiment.py       # Multi-task RoBERTa training
│   ├── train_reply.py           # FLAN-T5 + LoRA training
│   ├── data/                    # Sample labelled JSONL + data guide
│   ├── models/                  # Trained checkpoints land here (gitignored)
│   └── requirements.txt
├── README.md                    # Setup + quick-start
├── MODELS-GUIDE.md              # Detailed ML spec (this doc's ML twin)
├── context.md                   # (You are here)
├── package.json, tsconfig.json, next.config.ts
└── .env.example, .gitignore, LICENSE
```

### Adding a new feature

1. Design the pipeline — which of the 5 models do you need? Do you need
   a new endpoint on the FastAPI server or can you reuse existing ones?
2. Add the API route under `src/app/api/<feature>/route.ts`, chaining
   the models via `mlClient.ts`.
3. Add the page under `src/app/<feature>/page.tsx`, wrapped in
   `<DashboardShell>`.
4. Add a nav entry in `src/components/DashboardShell.tsx`.
5. If it saves data, add columns to `prisma/schema.prisma` and run
   `npx prisma db push`.

### Adding a new model

1. Write `ml/train_<model>.py` — follow the pattern in the existing
   training scripts (argparse, HF Trainer, save to `ml/models/<name>/`).
2. Add sample training data at `ml/data/<model>_labeled.jsonl` plus a
   line in `ml/data/README.md` for the schema.
3. Add an endpoint + stub + `_try_load_<model>` function to `ml/server.py`.
4. Add a corresponding function in `src/lib/mlClient.ts`.
5. Wire it into whatever route(s) need it.

### Type checking

```bash
npx tsc --noEmit --skipLibCheck
```

### Prisma studio (visual DB browser)

```bash
npx prisma studio
```

---

## 16. What's intentionally NOT here + why

- **No auth** — this is a local single-user demo. Adding auth adds
  complexity that doesn't demonstrate ML capability.
- **No billing** — same reason. The production repo has a full Freemius
  Merchant-of-Record integration.
- **No external API keys** — everything runs on local trained models.
  This is the entire point of the "local ML research build" framing.
- **No proxy for Reddit fetches** — at localhost demo scale, direct
  fetches from the developer machine work fine.
- **No 24/7 monitor cron** — this repo focuses on interactive research
  usage. Production has a `cron.js` process that runs saved monitors
  every 5 minutes.
- **No admin panel** — not relevant for a demo.
- **No marketing pages** — no pricing / sign-up / legal / contact. The
  home page is the entry point.
- **No mobile drawer polish** — a basic mobile-friendly layout is present
  but the production has more elaborate mobile UX.

Everything in this list COULD be added; the ML pipeline is designed to
be composable with more features layered on top.

---

## 17. Where to look for what

| Question | File |
|---|---|
| How does the Search pipeline work? | `src/app/api/search/route.ts` |
| How does Deep Scan mine comments? | `src/app/api/deep-scan/route.ts` |
| How does Discover rank subreddits? | `src/app/api/discover/route.ts` |
| How is each model architected? | `MODELS-GUIDE.md` |
| How do I train model X? | `ml/train_<x>.py` (docstring at the top) |
| What's the DB schema? | `prisma/schema.prisma` |
| How does the ML server work? | `ml/server.py` (top docstring) |
| What design tokens are available? | `src/app/globals.css` |
| Which model is used where? | `MODELS-GUIDE.md` → "Feature ↔ model matrix" |
| Reddit fetch helpers | `src/lib/reddit.ts` |
| HTTP client to ML server | `src/lib/mlClient.ts` |
| Sample training data schemas | `ml/data/README.md` |

---

## 18. Known trade-offs and future work

### Trade-offs

- **Bi-encoder vs cross-encoder for relevance** — chose bi-encoder for
  10-100× faster batch scoring at cost of ~3-5% accuracy on standard
  benchmarks. Right trade for real-time pipeline use.
- **LoRA vs full fine-tune for reply generator** — chose LoRA for
  affordable GPU training and simpler deployment (merged weights are
  a standard T5 checkpoint). Empirically matches full fine-tuning on
  constrained generation tasks.
- **DistilBERT vs RoBERTa for intent** — chose DistilBERT because it's
  the first stage of the pipeline (highest throughput requirement) and
  the intent task is relatively easy. RoBERTa would give ~2% better
  F1 at 2× the compute — not worth it for filter stage.
- **Rule-based stubs as fallback** — keeps the app runnable before
  training completes. Trade-off: stubs are much less accurate than
  trained models — teachers/graders should evaluate on trained
  checkpoints, not stub output.

### Future work

- **Cross-encoder re-ranker on top of the bi-encoder** — for the top-N
  post candidates, apply a slower but more accurate cross-encoder to
  refine the final ranking. Small end-to-end latency cost, meaningful
  precision improvement.
- **Adversarial fine-tuning for the role classifier** — augment
  training data with synthetically-generated hard examples (buyers
  phrased as sellers, sellers phrased ambiguously) to close the
  precision gap.
- **Continual learning** — retrain periodically on freshly-labelled
  data as Reddit language drifts (new products, new slang).
- **Larger reply generator** — FLAN-T5-large or FLAN-T5-XL with LoRA
  would produce noticeably higher-quality replies at 2-4× training cost.
- **Explicit sub-vs-supra-Reddit generalisation study** — measure how
  each model degrades on unseen subreddits and report per-subreddit
  F1 curves.

---

*Last updated with the addition of Deep Scan and Discover Subreddits
features. If a section here contradicts the actual code, the code wins —
please open an issue.*
