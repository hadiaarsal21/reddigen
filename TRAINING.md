# Training the ReddiGen models

End-to-end guide: build the datasets, train the five models on a free Kaggle
GPU, track every run with MLflow, and import the checkpoints back into the
local app.

For *what each model is and why*, see [MODELS-GUIDE.md](MODELS-GUIDE.md). This
document is the operational counterpart — the how.

---

## Contents

- [Why Kaggle](#why-kaggle)
- [Step 1 — Datasets](#step-1--datasets)
- [Step 2 — Train on Kaggle](#step-2--train-on-kaggle)
- [Step 3 — Import the checkpoints](#step-3--import-the-checkpoints)
- [Step 4 — Inspect the experiments](#step-4--inspect-the-experiments)
- [Training locally](#training-locally)
- [Troubleshooting](#troubleshooting)

---

## Why Kaggle

The five models total ~590M parameters. Two of them (RoBERTa-base) and the
FLAN-T5 reply generator are not practical to train on a CPU — the reply model
alone would take well over a day.

| Environment | Full training run |
|---|---|
| Kaggle T4 (free, 30 h/week) | **~1 hour** |
| Modern laptop CPU | 20+ hours |
| Intel HD Graphics iGPU | not usable — no CUDA |

Kaggle gives 30 GPU-hours per week at no cost, with 12-hour session limits, so
one sitting covers the whole run comfortably.

---

## Step 1 — Datasets

```bash
python ml/data/generate_dataset.py
```

Writes five JSONL files into `ml/data/`:

| File | Rows | Used by |
|---|---|---|
| `intent_labeled.jsonl` | 10,000 | Model 1 — intent |
| `relevance_pairs.jsonl` | 15,000 | Model 2 — relevance |
| `role_labeled.jsonl` | 8,000 | Model 3 — role |
| `sentiment_labeled.jsonl` | 12,000 | Model 4 — sentiment + urgency |
| `reply_pairs.jsonl` | 5,000 | Model 5 — reply |

Useful flags:

```bash
python ml/data/generate_dataset.py --scale 0.1   # 10% — fast rehearsal
python ml/data/generate_dataset.py --seed 7      # a different draw
```

Generation is deterministic for a given seed, so the corpus is reproducible.

### What this data is, and what it is not

It is **synthetic** — built compositionally from templates, domain vocabularies
and noise, not scraped from Reddit and not human-labelled.

That is a deliberate trade. Labelling 50,000 rows across five tasks is weeks of
annotator work, and Reddit's post-2023 API restrictions make bulk collection
impractical (the app itself has to fall back to RSS feeds because the JSON
endpoints return 403). Synthetic data lets the full pipeline — training,
tracking, export, serving — be built and validated now.

**Read the metrics accordingly.** A high macro-F1 on a held-out split of this
corpus shows the model learned *these templates*, not that it generalises to
live Reddit. The numbers validate that the pipeline works; they are not
publishable accuracy figures. `MODELS-GUIDE.md > Data acquisition` describes the
real labelling protocol — stratified sampling, two annotators, Cohen's kappa
≥ 0.7 — for when genuine data is available.

Swapping in real data needs no code changes: match the JSONL schema and every
training script reads it as-is.

---

## Step 2 — Train on Kaggle

**What you need to do:**

1. Go to [kaggle.com](https://www.kaggle.com) → sign in → **Create → New Notebook**
2. **File → Import Notebook** → upload `notebooks/reddigen_kaggle_training.ipynb`
3. Open the sidebar (**⋮ → Settings**) and set:
   - **Accelerator → GPU T4 x2** *(required — without it everything runs on CPU)*
   - **Internet → On** *(required — pip installs and HuggingFace downloads)*
4. If your GitHub repo is **private**, add a token so the notebook can clone it:
   **Add-ons → Secrets → Add secret**, name it exactly `GITHUB_TOKEN`, and paste
   a fine-grained PAT with *Contents: Read*. Skip this if the repo is public.
5. Edit the `GITHUB_USER` variable in cell 2 if you forked the repo.
6. **Run All**, then leave it — roughly an hour.
7. When it finishes, download **`reddigen-models.zip`** from the **Output** panel.

Kaggle disconnects idle browser sessions. For a long run use
**Save Version → Save & Run All (Commit)**, which executes headless and keeps the
output; you can close the tab.

### What the notebook does

| Cell | Purpose |
|---|---|
| 1 | Verifies a GPU is attached — stops early if not |
| 2 | Clones the repo (supports the private-repo token) |
| 3 | Installs deps, skipping torch (Kaggle's build is already CUDA-matched) |
| 4 | Generates the datasets and prints class balance |
| 5 | Points MLflow at a local SQLite store |
| 6 | Trains all five models, continuing past any single failure |
| 7 | Prints the tracked runs and metrics |
| 8 | Loads a checkpoint back and runs a sample prediction |
| 9 | Packages `reddigen-models.zip` |

Expected timings on a T4: intent ~6 min, relevance ~4, role ~12,
sentiment ~15, reply ~20.

---

## Step 3 — Import the checkpoints

From the repo root:

```bash
python ml/import_models.py ~/Downloads/reddigen-models.zip
```

This unpacks into `ml/models/<name>/`, skipping the `checkpoint-*` folders that
carry optimizer state (they are several times the size of the weights and
inference never reads them).

Verify:

```bash
curl http://localhost:8000/
```

Every entry under `models_loaded` should now read `true`:

```json
{"ok":true,"models_loaded":{"intent":true,"relevance":true,"role":true,"sentiment":true,"reply":true}}
```

**No restart is required** — `server.py` loads each checkpoint lazily on the
first request that needs it. The one exception is *removing* a model: the loaded
pipeline stays cached in memory, so reverting to stubs does need a restart.

Use `--force` to overwrite checkpoints that are already installed.

---

## Step 4 — Inspect the experiments

Every run records hyperparameters, dataset size and class balance, the hardware
it ran on, per-epoch metrics, final evaluation, and checkpoint size.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open <http://localhost:5000>. If you unpacked `mlflow.db` from the Kaggle
zip, the Kaggle runs appear here too.

Experiments are one per model: `reddigen-intent`, `reddigen-relevance`,
`reddigen-role`, `reddigen-sentiment-urgency`, `reddigen-reply`.

Point runs at a remote tracking server instead by exporting `MLFLOW_TRACKING_URI`
before training.

Two behaviours worth knowing:

- The default backend is **SQLite**, not the classic `./mlruns` directory —
  MLflow 3.x put the plain-file store into maintenance mode and errors out
  unless you opt in. A directory path still works; the helper sets
  `MLFLOW_ALLOW_FILE_STORE` for you.
- Setting `MLFLOW_LOG_CHECKPOINTS=false` records checkpoint *sizes* without
  copying ~2.5 GB of weights into the artifact store. The Kaggle notebook sets
  this, since it exports the weights separately.

---

## Training locally

Possible without a GPU, but only sensibly at reduced scale:

```bash
python ml/data/generate_dataset.py --scale 0.05
python ml/train_intent.py --epochs 1 --batch-size 8
```

Every script takes `--data`, `--out`, `--epochs`, `--batch-size` and `--lr`, so
nothing is hard-coded to Kaggle.

---

## Troubleshooting

**`CUDA available: False` in the notebook**
The accelerator is not attached. Settings → Accelerator → GPU T4 x2, then Run All.

**`CUDA out of memory`**
Halve the batch size for that job in cell 6. The reply model is the hungriest —
try `--batch-size 4`.

**Clone fails with `could not read Username`**
The repo is private and no `GITHUB_TOKEN` secret is set. Add one (Add-ons →
Secrets) or make the repo public.

**`Trainer.__init__() got an unexpected keyword argument 'tokenizer'`**
transformers < 4.46. The scripts use the current `processing_class` argument;
upgrade with `pip install -U "transformers>=4.46"`.

**`Unexpected UTF-8 BOM` when loading JSONL**
A Windows editor saved the file with a byte-order mark. The loaders use
`utf-8-sig` and tolerate this; if you wrote the file with a custom script, save
it as UTF-8 without BOM.

**Kaggle session disconnects mid-run**
Use Save Version → Save & Run All (Commit) for headless execution.

**Models load but predictions look random**
Check the row count the training script printed. A few hundred rows for one
epoch will not converge — that is a smoke test, not a trained model. Use the
full corpus and the notebook's default epochs.
