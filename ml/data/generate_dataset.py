"""
Synthetic dataset generator for all five ReddiGen models.

Why synthetic? Labelling real Reddit data at the volume the models need
(5K-15K rows each, see MODELS-GUIDE.md) costs weeks of annotator time, and
Reddit's post-2023 API restrictions make bulk collection impractical. This
generator produces a compositional corpus — templates x domains x modifiers
x noise — that is large, balanced, and reproducible, so the training
pipeline can be exercised end to end and the architectures validated.

What it is good for:
  - validating that every training script runs, converges and exports
  - benchmarking throughput and GPU memory on Kaggle
  - demonstrating the full pipeline with real checkpoints instead of stubs

What it is NOT:
  - a substitute for human-labelled Reddit data. Scores on a held-out split
    of this corpus measure how well a model learned THESE templates, not how
    well it generalises to live Reddit. Treat the numbers as pipeline
    smoke-tests, not published accuracy. See MODELS-GUIDE.md > Data
    acquisition for the real labelling protocol.

Usage:
    python ml/data/generate_dataset.py                 # default sizes
    python ml/data/generate_dataset.py --scale 0.1     # 10% for a quick run
    python ml/data/generate_dataset.py --seed 7        # different draw
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Vocabulary ──────────────────────────────────────────────────────────

SERVICES = [
    "python developer", "SEO agency", "video editor", "copywriter",
    "web designer", "data analyst", "virtual assistant", "logo designer",
    "shopify expert", "wordpress developer", "social media manager",
    "bookkeeper", "email marketer", "UI/UX designer", "mobile app developer",
    "content writer", "ads specialist", "react developer", "devops engineer",
    "technical recruiter", "product photographer", "voice over artist",
    "translator", "brand strategist", "automation consultant",
    "growth marketer", "penetration tester", "database administrator",
    "3D modeller", "podcast editor",
]

CONTEXTS = [
    "SaaS startup", "shopify store", "youtube channel", "small business",
    "agency", "side project", "e-commerce brand", "local restaurant",
    "fitness app", "newsletter", "online course", "nonprofit",
    "indie game", "real estate business", "consulting firm", "dental clinic",
    "coffee roastery", "law practice", "photography studio", "SaaS product",
]

TASKS = [
    "set up analytics", "improve page speed", "rank on Google",
    "automate invoicing", "migrate to a new host", "reduce churn",
    "write better landing copy", "run Facebook ads", "build a CI pipeline",
    "clean up my database", "design a logo", "edit my videos faster",
    "grow my email list", "handle GDPR compliance", "scale my backend",
    "set up a CRM", "optimise checkout", "structure my pricing",
]

TOPICS = [
    "the new API pricing", "remote work", "AI replacing junior devs",
    "the latest framework release", "freelance rates in 2026",
    "this subreddit's rules", "open source burnout", "tech layoffs",
    "no-code tools", "the state of SEO", "hustle culture",
    "junior hiring standards", "subscription fatigue", "crypto payments",
]

OFF_TOPIC = [
    "My cat just knocked over my monitor for the third time today",
    "Finally beat the final boss after 40 hours, AMA",
    "Does anyone else hate Mondays or is it just me",
    "Rate my desk setup — took two years to get here",
    "Happy birthday to my dog, he is 7 today",
    "Just wanted to say this community is great",
    "Found this weird bug in a game from 2003",
    "Anyone watching the match tonight",
    "My sourdough starter finally worked",
    "Traffic was insane this morning, took me 2 hours",
    "Upvote if you are also procrastinating right now",
    "Reminder to drink water everyone",
]

# Substantive off-topic bodies. Without these the class was the only one that
# usually had an empty body, and the classifier learned "no body => off_topic"
# instead of learning intent — which broke it on Reddit link posts, whose
# selftext is always empty.
OFF_TOPIC_BODIES = [
    "Third time this week. I have started taping the cable down.",
    "Took way longer than I expected but worth every hour.",
    "Not sure why I am even posting this, just needed to vent a bit.",
    "Been lurking here for years, first time actually posting.",
    "Photos in the comments if anyone is curious.",
    "Sorry if this is the wrong sub, mods feel free to remove.",
    "Anyway, hope everyone is having a decent week.",
    "My partner thinks I am being ridiculous about this.",
    "Genuinely one of the highlights of my month, which says a lot.",
    "Edit: wow this blew up, thanks for the awards.",
]

# Probability that a post has no body at all, applied UNIFORMLY across every
# label so body presence carries no signal about the class. Reddit link posts
# have empty selftext regardless of intent, so the models must cope with it.
EMPTY_BODY_RATE = 0.22

BUDGETS = [
    "budget is around $500", "can pay $50/hr", "budget $2k for the project",
    "looking to spend under $1000", "have a $300 monthly budget",
    "flexible on budget for the right person", "budget is tight, maybe $200",
    "ready to pay upfront", "$5k for a 3 month engagement",
]

URGENCY_HIGH = [
    "Need this done by Friday", "This is urgent", "ASAP please",
    "We go live Monday", "Deadline is tomorrow", "Client is waiting",
    "Site is down right now", "Losing money every day this is broken",
]
URGENCY_MED = [
    "Hoping to start in a couple of weeks", "Sometime this month works",
    "No massive rush but soon", "Would like this sorted before Q3",
]
URGENCY_LOW = [
    "No rush at all", "Just planning ahead for next year",
    "Exploring options for now", "Whenever you have time",
]

POSITIVE = [
    "Really happy with how things are going", "This community has been great",
    "Excited to get started", "Things are finally clicking",
    "Thanks in advance, you all rock",
]
NEGATIVE = [
    "Honestly getting really frustrated", "This has been a nightmare",
    "Wasted three months on the last guy", "I am so done with this",
    "Absolutely fed up at this point", "Worst experience I have had",
]
NEUTRAL = [
    "Just looking at options", "Here is where things stand",
    "Some background on the situation", "Posting to see what people think",
]

TONES = ["helpful", "professional", "casual", "empathetic"]

# Tasks that plausibly belong to a given service. Drawing the two
# independently produces nonsense like "content writer -> build a CI
# pipeline", which teaches the models spurious associations.
SERVICE_TASKS = {
    "python developer": ["automate invoicing", "clean up my database", "scale my backend"],
    "SEO agency": ["rank on Google", "improve page speed", "write better landing copy"],
    "video editor": ["edit my videos faster"],
    "copywriter": ["write better landing copy", "structure my pricing"],
    "web designer": ["optimise checkout", "improve page speed"],
    "data analyst": ["set up analytics", "clean up my database"],
    "virtual assistant": ["automate invoicing", "set up a CRM"],
    "logo designer": ["design a logo"],
    "shopify expert": ["optimise checkout", "migrate to a new host"],
    "wordpress developer": ["migrate to a new host", "improve page speed"],
    "social media manager": ["grow my email list", "run Facebook ads"],
    "bookkeeper": ["automate invoicing"],
    "email marketer": ["grow my email list", "reduce churn"],
    "UI/UX designer": ["optimise checkout", "design a logo"],
    "mobile app developer": ["scale my backend"],
    "content writer": ["write better landing copy", "grow my email list"],
    "ads specialist": ["run Facebook ads"],
    "react developer": ["improve page speed", "optimise checkout"],
    "devops engineer": ["build a CI pipeline", "scale my backend", "migrate to a new host"],
    "database administrator": ["clean up my database", "scale my backend"],
    "growth marketer": ["reduce churn", "grow my email list", "structure my pricing"],
    "automation consultant": ["automate invoicing", "set up a CRM"],
    "penetration tester": ["handle GDPR compliance"],
}


def task_for(rng: random.Random, service: str) -> str:
    """Pick a task that makes sense for this service, else any task."""
    return rng.choice(SERVICE_TASKS.get(service, TASKS))


# ── Helpers ─────────────────────────────────────────────────────────────

def maybe(rng: random.Random, text: str, p: float = 0.5) -> str:
    """Include `text` with probability p, else empty string."""
    return text if rng.random() < p else ""


def join_body(rng: random.Random, *parts: str) -> str:
    """Join non-empty parts into a body with natural spacing."""
    chunks = [p.strip() for p in parts if p and p.strip()]
    if not chunks:
        return ""
    sep = "\n\n" if rng.random() < 0.35 else " "
    return sep.join(chunks)


def maybe_drop_body(rng: random.Random, body: str) -> str:
    """
    Blank the body at a fixed rate, identically for every label.

    Body presence must not correlate with the class, or the model learns that
    shortcut instead of the task.
    """
    return "" if rng.random() < EMPTY_BODY_RATE else body


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>6,} rows -> {path}")


# ── Model 1: intent ─────────────────────────────────────────────────────

def gen_intent(rng: random.Random, n: int) -> list[dict]:
    rows = []
    per = n // 4

    for _ in range(per):  # buying_intent
        s, c = rng.choice(SERVICES), rng.choice(CONTEXTS)
        title = rng.choice([
            f"Looking for a {s} for my {c}",
            f"Need to hire a {s}",
            f"Anyone recommend a good {s}?",
            f"[Hiring] {s} for {c}",
            f"Where can I find a reliable {s}?",
            f"Paying for a {s} — recommendations?",
            f"Searching for {s} to help with my {c}",
        ])
        body = join_body(
            rng,
            f"Running a {c} and need help.",
            maybe(rng, rng.choice(BUDGETS), 0.7),
            maybe(rng, rng.choice(URGENCY_HIGH + URGENCY_MED), 0.5),
            maybe(rng, "DM me with your portfolio.", 0.4),
        )
        body = maybe_drop_body(rng, body)
        rows.append({"text": f"{title}\n\n{body}", "label": "buying_intent"})

    for _ in range(per):  # advice_seeking
        t, c = rng.choice(TASKS), rng.choice(CONTEXTS)
        title = rng.choice([
            f"How do I {t} for my {c}?",
            f"What's the best way to {t}?",
            f"Struggling to {t} — any tips?",
            f"Best approach to {t} in 2026?",
            f"Can someone explain how to {t}?",
        ])
        body = join_body(
            rng,
            f"I want to do this myself rather than hire someone.",
            maybe(rng, "Tried a few tutorials but they are outdated.", 0.6),
            maybe(rng, "Any guides appreciated.", 0.4),
        )
        body = maybe_drop_body(rng, body)
        rows.append({"text": f"{title}\n\n{body}", "label": "advice_seeking"})

    for _ in range(per):  # discussion
        topic = rng.choice(TOPICS)
        title = rng.choice([
            f"Thoughts on {topic}?",
            f"Unpopular opinion: {topic} is overrated",
            f"Is anyone else concerned about {topic}?",
            f"Let's talk about {topic}",
            f"{topic} — where do you all stand?",
        ])
        body = join_body(
            rng,
            "Curious what this community thinks.",
            maybe(rng, "I have gone back and forth on this for a while.", 0.5),
        )
        body = maybe_drop_body(rng, body)
        rows.append({"text": f"{title}\n\n{body}", "label": "discussion"})

    for _ in range(n - 3 * per):  # off_topic
        title = rng.choice(OFF_TOPIC)
        body = join_body(
            rng,
            rng.choice(OFF_TOPIC_BODIES),
            maybe(rng, rng.choice(OFF_TOPIC_BODIES), 0.3),
        )
        body = maybe_drop_body(rng, body)
        rows.append({"text": f"{title}\n\n{body}", "label": "off_topic"})

    rng.shuffle(rows)
    return rows


# ── Model 2: relevance pairs ────────────────────────────────────────────

def gen_relevance(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        s, c = rng.choice(SERVICES), rng.choice(CONTEXTS)
        query = rng.choice([s, f"{s} for {c}", f"{s} services", f"hire {s}"])
        title = rng.choice([
            f"Looking for a {s} for my {c}",
            f"Need a {s} — any recommendations?",
            f"[Hiring] {s} for a {c}",
            f"Can anyone point me to a solid {s}?",
        ])
        body = join_body(
            rng,
            f"We run a {c} and need {s} support.",
            maybe(rng, rng.choice(BUDGETS), 0.6),
        )
        rows.append({"query": query, "positive": f"{title}\n\n{body}"})
    return rows


# ── Model 3: role ───────────────────────────────────────────────────────

def gen_role(rng: random.Random, n: int) -> list[dict]:
    rows = []
    # buyers deliberately over-sampled vs the ~15% real rate so the focal
    # loss has enough minority signal to learn from
    weights = {"buyer": 0.30, "seller": 0.30, "advisor": 0.25, "other": 0.15}

    for _ in range(n):
        role = rng.choices(list(weights), weights=list(weights.values()))[0]
        s = rng.choice(SERVICES)
        if role == "buyer":
            text = rng.choice([
                f"I need this too! Do you have availability for a {s}?",
                f"How much do you charge? Looking for exactly this.",
                f"Same boat here — been hunting for a {s} for weeks. DMing you.",
                f"Are you taking new clients? I need {s} work done.",
                f"What's your rate? I have budget approved already.",
                f"Can you do this for a smaller project too? Interested.",
            ])
        elif role == "seller":
            text = rng.choice([
                f"I offer exactly this — DM me for pricing and portfolio.",
                f"I've been a {s} for 6 years, happy to help. Sent you a message.",
                f"Check my profile, I do {s} work full time.",
                f"Available now. My rate is competitive, see pinned post.",
                f"I run an agency that specialises in this. Let's talk.",
            ])
        elif role == "advisor":
            text = rng.choice([
                f"You should hire someone with at least 5 years experience.",
                f"Make sure you ask for references before paying anything.",
                f"Honestly you could learn this yourself in a weekend.",
                f"Watch out for people who ask for full payment upfront.",
                f"I'd suggest posting in a more specialised subreddit.",
            ])
        else:
            text = rng.choice([
                "lol same", "RemindMe! 3 days", "Following this thread",
                "Commenting so I can find this later", "This is the way",
                "Wrong sub for this I think", "Good luck OP",
            ])
        rows.append({"text": text, "label": role})
    rng.shuffle(rows)
    return rows


# ── Model 4: sentiment + urgency ────────────────────────────────────────

def gen_sentiment(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        sentiment = rng.choices(
            ["positive", "neutral", "negative"], weights=[0.3, 0.4, 0.3]
        )[0]
        # 'high' slightly over-represented per MODELS-GUIDE
        urgency = rng.choices(["low", "medium", "high"], weights=[0.3, 0.3, 0.4])[0]

        sent_txt = {
            "positive": rng.choice(POSITIVE),
            "neutral": rng.choice(NEUTRAL),
            "negative": rng.choice(NEGATIVE),
        }[sentiment]
        urg_txt = {
            "low": rng.choice(URGENCY_LOW),
            "medium": rng.choice(URGENCY_MED),
            "high": rng.choice(URGENCY_HIGH),
        }[urgency]

        s, c = rng.choice(SERVICES), rng.choice(CONTEXTS)
        base = rng.choice([
            f"Trying to find a {s} for our {c}.",
            f"We need {s} help on the {c}.",
            f"Been looking into {s} options for the {c}.",
        ])
        text = join_body(rng, base, sent_txt, urg_txt)
        rows.append({"text": text, "sentiment": sentiment, "urgency": urgency})
    rng.shuffle(rows)
    return rows


# ── Model 5: reply pairs ────────────────────────────────────────────────

REPLY_TEMPLATES = {
    "helpful": [
        "Hey! I've helped a few {ctx} with exactly this. Happy to share what worked — "
        "the biggest win is usually {task}. Feel free to DM if you want specifics.",
        "I do {svc} work and this comes up a lot. Quick tip: start with {task}. "
        "Glad to talk it through if useful.",
    ],
    "professional": [
        "Hi — I specialise in {svc} for {ctx}. I've delivered similar projects and "
        "can share relevant case studies. Happy to arrange a short call to scope this out.",
        "Good afternoon. This is squarely in my area ({svc}). I'd suggest beginning with "
        "{task}. I can send over a proposal if that would be helpful.",
    ],
    "casual": [
        "oh this is right up my alley — been doing {svc} stuff for {ctx} for a while. "
        "honestly {task} is where I'd start. give me a shout if you want a hand",
        "hey! I do this kind of thing. {task} first, everything else after. dm me if you want",
    ],
    "empathetic": [
        "That sounds really frustrating — a lot of people hit the same wall with their {ctx}. "
        "You're not doing anything wrong. If it helps, {task} usually unblocks things. "
        "Happy to help however I can.",
        "I know how draining this gets. I've worked with {ctx} in the same spot. "
        "Start with {task} and it gets much more manageable. Here if you need a hand.",
    ],
}


def gen_reply(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        s, c = rng.choice(SERVICES), rng.choice(CONTEXTS)
        t = task_for(rng, s)
        tone = rng.choice(TONES)
        title = rng.choice([
            f"Looking for a {s} for my {c}",
            f"Need help — {t} for my {c}",
            f"[Hiring] {s}",
            f"Anyone know a good {s}?",
        ])
        body = join_body(
            rng,
            f"We run a {c} and need to {t}.",
            maybe(rng, rng.choice(BUDGETS), 0.6),
            maybe(rng, rng.choice(URGENCY_HIGH + URGENCY_MED + URGENCY_LOW), 0.5),
        )
        reply = rng.choice(REPLY_TEMPLATES[tone]).format(svc=s, ctx=c, task=t)
        rows.append({
            "query": f"{s} for {c}",
            "post_title": title,
            "post_body": body,
            "tone": tone,
            "reply": reply,
        })
    return rows


# ── Entry point ─────────────────────────────────────────────────────────

DEFAULT_SIZES = {
    "intent": 10_000,
    "relevance": 15_000,
    "role": 8_000,
    "sentiment": 12_000,
    "reply": 5_000,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="ml/data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--scale", type=float, default=1.0,
        help="Multiply all dataset sizes (0.1 = 10%% for a fast smoke run)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    sizes = {k: max(20, int(v * args.scale)) for k, v in DEFAULT_SIZES.items()}

    print(f"Generating datasets (seed={args.seed}, scale={args.scale})")
    write_jsonl(out / "intent_labeled.jsonl", gen_intent(rng, sizes["intent"]))
    write_jsonl(out / "relevance_pairs.jsonl", gen_relevance(rng, sizes["relevance"]))
    write_jsonl(out / "role_labeled.jsonl", gen_role(rng, sizes["role"]))
    write_jsonl(out / "sentiment_labeled.jsonl", gen_sentiment(rng, sizes["sentiment"]))
    write_jsonl(out / "reply_pairs.jsonl", gen_reply(rng, sizes["reply"]))
    print("Done.")


if __name__ == "__main__":
    main()
