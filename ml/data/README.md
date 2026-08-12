# Training Data

Each of the five training scripts reads a labelled JSONL file from this
directory. The sample files below are small starter sets (~50 examples
each) so you can smoke-test the whole training pipeline end-to-end on a
CPU in a few minutes. For real training, expand to at least 5,000 rows
per task by (a) manually labelling posts you sample from Reddit's public
archives, (b) using a stronger model to bootstrap silver labels, or (c)
crowd-annotating via Prolific or a similar service.

| Script | File | Format per row |
|---|---|---|
| `train_intent.py` | `intent_labeled.jsonl` | `{"text": "title\\n\\nbody", "label": "buying_intent"\|"advice_seeking"\|"discussion"\|"off_topic"}` |
| `train_relevance.py` | `relevance_pairs.jsonl` | `{"query": "SEO for SaaS", "positive": "title\\n\\nbody"}` |
| `train_role.py` | `role_labeled.jsonl` | `{"text": "comment body", "label": "buyer"\|"seller"\|"advisor"\|"other"}` |
| `train_sentiment.py` | `sentiment_labeled.jsonl` | `{"text": "...", "sentiment": "positive"\|"neutral"\|"negative", "urgency": "low"\|"medium"\|"high"}` |
| `train_reply.py` | `reply_pairs.jsonl` | `{"query": "...", "post_title": "...", "post_body": "...", "tone": "helpful", "reply": "the gold reply"}` |

## Suggested public data sources

- **Reddit historical archives** — the [Pushshift](https://pushshift.io/)
  monthly dumps are the canonical source for training-scale Reddit corpora.
  Filter to subreddits relevant to your use-case (e.g. r/SaaS, r/Entrepreneur,
  r/marketing, r/webdev).
- **Sentiment pre-training** — [Sentiment140](http://help.sentiment140.com/for-students)
  (1.6M tweets, coarse polarity) or [SemEval-2017 Task 4](https://alt.qcri.org/semeval2017/task4/)
  for a warm start before fine-tuning on Reddit.
- **General-purpose classification** — [Amazon Reviews](https://nijianmo.github.io/amazon/index.html)
  transfers surprisingly well to intent-detection tasks after adapter tuning.

## Data-quality tips

- **Inter-annotator agreement**: label a 500-row overlap set with two
  annotators and require Cohen's kappa >= 0.7 before trusting the rest.
- **Class balance**: `train_role.py` handles imbalance internally with
  focal loss and class weights, but you still need at least ~200 examples
  of the minority class (`buyer`) for the model to learn it stably.
- **Domain overlap between train and test splits**: split by SUBREDDIT
  rather than by row so the test set contains subreddits the model has
  never seen — the reported metric then reflects real-world generalisation.
