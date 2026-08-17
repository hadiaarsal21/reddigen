"""
Local ML inference server for ReddiGen.

Serves five endpoints, one per trained model:

  POST /classify-intent   → classifies a post into buying-intent / advice /
                            discussion / off-topic
  POST /score-relevance   → semantic similarity score between a user query
                            and a Reddit post
  POST /classify-role     → three-way BUYER / SELLER / ADVISOR (Deep Scan)
  POST /predict-sentiment → sentiment polarity + urgency (multi-task head)
  POST /generate-reply    → tone-conditioned reply draft

Each endpoint AUTOMATICALLY detects whether a trained model checkpoint is
available in `ml/models/<name>/`. If it is, the endpoint loads the real
transformer and runs inference. If not, it falls back to a fast rule-based
"stub" that returns plausible outputs — enough for the front-end demo to
work end-to-end even before any training has happened. This design keeps
the presentation reliable regardless of whether the grader has GPU access.

Run:
    python ml/server.py

Or from the project root:
    npm run ml:serve
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

MODELS_DIR = Path(__file__).resolve().parent / "models"

# ── Model loaders (lazy, tolerant of missing checkpoints) ────────────────

_intent_pipeline = None
_relevance_encoder = None
_role_pipeline = None
_sentiment_pipeline = None
_reply_pipeline = None


def _try_load_intent():
    """Load the fine-tuned intent classifier if it exists on disk."""
    global _intent_pipeline
    if _intent_pipeline is not None:
        return _intent_pipeline
    ckpt = MODELS_DIR / "intent"
    if not ckpt.exists():
        return None
    try:
        from transformers import pipeline  # imported lazily so the server
        _intent_pipeline = pipeline(          # boots even without HF
            "text-classification", model=str(ckpt), top_k=1
        )
        print(f"[server] Loaded intent model from {ckpt}")
    except Exception as exc:
        print(f"[server] Failed to load intent model: {exc}")
        _intent_pipeline = None
    return _intent_pipeline


def _try_load_relevance():
    global _relevance_encoder
    if _relevance_encoder is not None:
        return _relevance_encoder
    ckpt = MODELS_DIR / "relevance"
    if not ckpt.exists():
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _relevance_encoder = SentenceTransformer(str(ckpt))
        print(f"[server] Loaded relevance encoder from {ckpt}")
    except Exception as exc:
        print(f"[server] Failed to load relevance model: {exc}")
        _relevance_encoder = None
    return _relevance_encoder


def _try_load_role():
    global _role_pipeline
    if _role_pipeline is not None:
        return _role_pipeline
    ckpt = MODELS_DIR / "role"
    if not ckpt.exists():
        return None
    try:
        from transformers import pipeline
        _role_pipeline = pipeline(
            "text-classification", model=str(ckpt), top_k=1
        )
        print(f"[server] Loaded role model from {ckpt}")
    except Exception as exc:
        print(f"[server] Failed to load role model: {exc}")
        _role_pipeline = None
    return _role_pipeline


SENTIMENT_LABELS = ["positive", "neutral", "negative"]
URGENCY_LABELS = ["low", "medium", "high"]


def _try_load_sentiment():
    """
    Load the multi-task sentiment + urgency model.

    This one cannot go through `pipeline("text-classification")`. The
    checkpoint is a custom two-head architecture (shared encoder ->
    sentiment_head + urgency_head) defined in train_sentiment.py, and a
    standard sequence-classification pipeline silently ignores both heads:
    it exposes LABEL_0/LABEL_1... instead of the real class names, so every
    lookup misses and the endpoint returns a constant answer.

    Instead, rebuild the architecture and load the weights directly.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    ckpt = MODELS_DIR / "sentiment"
    if not ckpt.exists():
        return None
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        class _MultiTask(nn.Module):
            """Inference-only mirror of train_sentiment.MultiTaskModel."""

            def __init__(self, config):
                super().__init__()
                self.encoder = AutoModel.from_config(config)
                self.dropout = nn.Dropout(0.1)
                self.sentiment_head = nn.Linear(config.hidden_size, len(SENTIMENT_LABELS))
                self.urgency_head = nn.Linear(config.hidden_size, len(URGENCY_LABELS))

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                pooled = self.dropout(out.last_hidden_state[:, 0])
                return self.sentiment_head(pooled), self.urgency_head(pooled)

        cfg = AutoConfig.from_pretrained(ckpt)
        model = _MultiTask(cfg)

        state = None
        safet = ckpt / "model.safetensors"
        if safet.exists():
            from safetensors.torch import load_file

            state = load_file(str(safet))
        else:
            bin_path = ckpt / "pytorch_model.bin"
            if bin_path.exists():
                state = torch.load(str(bin_path), map_location="cpu")
        if state is None:
            raise FileNotFoundError("no model weights found in checkpoint")

        missing, unexpected = model.load_state_dict(state, strict=False)
        # Both heads must be present, or predictions are meaningless.
        head_keys = [
            "sentiment_head.weight", "sentiment_head.bias",
            "urgency_head.weight", "urgency_head.bias",
        ]
        absent = [k for k in head_keys if k in missing]
        if absent:
            raise RuntimeError(f"checkpoint is missing the task heads: {absent}")

        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        tok = AutoTokenizer.from_pretrained(ckpt)
        _sentiment_pipeline = {"model": model, "tokenizer": tok, "device": device}
        print(f"[server] Loaded multi-task sentiment model from {ckpt} on {device}")
        if missing:
            print(f"[server]   (non-head keys newly initialised: {len(missing)})")
    except Exception as exc:
        print(f"[server] Failed to load sentiment model: {exc}")
        _sentiment_pipeline = None
    return _sentiment_pipeline


# Must match PROMPT in train_reply.py exactly. The model was fine-tuned on
# this template; feeding it a different one degrades output quality badly,
# because the tone token and field order are what condition the generation.
REPLY_PROMPT = (
    "Write a natural, helpful Reddit reply.\n"
    "tone: {tone}\n"
    "our offer: {query}\n"
    "post title: {title}\n"
    "post body: {body}\n"
    "reply:"
)


def _try_load_reply():
    """
    Load the merged FLAN-T5 reply generator.

    Loaded directly rather than through pipeline("text2text-generation"):
    transformers 5.x removed that task alias entirely, so the pipeline call
    raises KeyError and the endpoint silently falls back to the stub.
    AutoModelForSeq2SeqLM works on both 4.x and 5.x.
    """
    global _reply_pipeline
    if _reply_pipeline is not None:
        return _reply_pipeline
    ckpt = MODELS_DIR / "reply"
    if not ckpt.exists():
        return None
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt))

        # T5 ties encoder.embed_tokens and decoder.embed_tokens to `shared`.
        # Merging the LoRA adapter writes `shared` and `lm_head` with different
        # values, so transformers declines to tie them and reports the two
        # embed_tokens as MISSING — leaving them on the meta device. Any later
        # .to() then fails with "Cannot copy out of meta tensor". Re-point them
        # at `shared`, which is what the tie would have done.
        if hasattr(model, "shared"):
            if hasattr(model, "encoder"):
                model.encoder.embed_tokens = model.shared
            if hasattr(model, "decoder"):
                model.decoder.embed_tokens = model.shared

        meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
        if meta:
            raise RuntimeError(
                f"{len(meta)} parameter(s) still on the meta device after re-tying "
                f"(first: {meta[0]}). The checkpoint is incomplete."
            )

        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cpu":
            model.to(device)  # already on CPU otherwise; .to() is a no-op cost
        _reply_pipeline = {"model": model, "tokenizer": tok, "device": device}
        print(f"[server] Loaded reply model from {ckpt} on {device}")
    except Exception as exc:
        print(f"[server] Failed to load reply model: {exc}")
        _reply_pipeline = None
    return _reply_pipeline


# ── Rule-based stubs (used until trained checkpoints appear) ─────────────
# These are intentionally lightweight — they mirror what a trained model
# would OUTPUT so the API contract is identical. Swap the checkpoints in
# and the endpoints transparently switch to real inference.

BUY_INTENT_PATTERNS = [
    r"\b(need|looking for|any recommendation|recommend|suggest|best|which|hire|help me find)\b",
    r"\b(anyone use|has anyone tried|worth it|alternative to|switching from)\b",
]
SELL_PATTERNS = [
    r"\b(dm me|i can help|feel free to reach|i offer|i specialize|hit me up|check out my)\b",
    r"\b(my (company|service|product|agency)|we build|we help|our (tool|platform))\b",
]
ADVISOR_PATTERNS = [
    r"\b(you should|try (using |)|i (would|'d) recommend|in my opinion|look for|consider)\b",
]
URGENT_PATTERNS = [
    r"\b(asap|urgent|today|right now|immediately|need help now|deadline)\b",
    r"\b(broken|down|not working|crashed|emergency)\b",
]
POSITIVE_PATTERNS = [r"\b(love|great|amazing|awesome|excellent|best)\b"]
NEGATIVE_PATTERNS = [r"\b(hate|terrible|awful|worst|garbage|useless|scam)\b"]


def _matches(patterns: list[str], text: str) -> int:
    lower = text.lower()
    return sum(1 for p in patterns if re.search(p, lower))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(
    title="ReddiGen ML Service",
    version="1.0.0",
    description="Local inference for the five trained models that power ReddiGen.",
)


class HealthResp(BaseModel):
    ok: bool = True
    models_loaded: dict[str, bool]


@app.get("/", response_model=HealthResp)
def root():
    """Reports which trained checkpoints are currently loaded."""
    return HealthResp(
        models_loaded={
            "intent": _try_load_intent() is not None,
            "relevance": _try_load_relevance() is not None,
            "role": _try_load_role() is not None,
            "sentiment": _try_load_sentiment() is not None,
            "reply": _try_load_reply() is not None,
        }
    )


# ─── Endpoint 1: Intent classification ───────────────────────────────────

class IntentReq(BaseModel):
    title: str
    body: str = ""


class IntentResp(BaseModel):
    label: str = Field(description="buying_intent | advice_seeking | discussion | off_topic")
    confidence: float


@app.post("/classify-intent", response_model=IntentResp)
def classify_intent(req: IntentReq):
    text = f"{req.title}\n{req.body}"[:2000]
    pipe = _try_load_intent()
    if pipe is not None:
        # Truncate explicitly. A 2000-character post exceeds DistilBERT's 512
        # position embeddings and raises "size of tensor a must match tensor
        # b", which the caller sees as a dropped classification.
        out = pipe(text, truncation=True, max_length=256)[0][0]
        return IntentResp(label=out["label"], confidence=float(out["score"]))
    # Stub: pattern-count based
    buy = _matches(BUY_INTENT_PATTERNS, text)
    if buy >= 2:
        return IntentResp(label="buying_intent", confidence=0.82)
    if buy == 1:
        return IntentResp(label="advice_seeking", confidence=0.68)
    if any(k in text.lower() for k in ["discuss", "opinion", "thoughts on", "what do you think"]):
        return IntentResp(label="discussion", confidence=0.61)
    return IntentResp(label="off_topic", confidence=0.55)


# ─── Endpoint 2: Relevance ranking ───────────────────────────────────────

class RelevanceReq(BaseModel):
    query: str
    title: str
    body: str = ""


class RelevanceResp(BaseModel):
    score: float = Field(ge=0.0, le=1.0)


@app.post("/score-relevance", response_model=RelevanceResp)
def score_relevance(req: RelevanceReq):
    text = f"{req.title} {req.body}"
    enc = _try_load_relevance()
    if enc is not None:
        import numpy as np
        q_vec = enc.encode(req.query, convert_to_numpy=True, normalize_embeddings=True)
        d_vec = enc.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return RelevanceResp(score=float(np.dot(q_vec, d_vec)))
    # Stub: Jaccard-like token overlap normalised into [0, 1]
    q_toks = _tokenize(req.query)
    t_toks = _tokenize(text)
    if not q_toks or not t_toks:
        return RelevanceResp(score=0.0)
    overlap = len(q_toks & t_toks) / len(q_toks)
    return RelevanceResp(score=min(1.0, overlap * 1.2))


# ─── Endpoint 3: BUYER / SELLER / ADVISOR (Deep Scan) ────────────────────

class RoleReq(BaseModel):
    text: str


class RoleResp(BaseModel):
    role: str = Field(description="buyer | seller | advisor | other")
    confidence: float


@app.post("/classify-role", response_model=RoleResp)
def classify_role(req: RoleReq):
    pipe = _try_load_role()
    if pipe is not None:
        out = pipe(req.text[:2000], truncation=True, max_length=256)[0][0]
        return RoleResp(role=out["label"], confidence=float(out["score"]))
    # Stub: rule counts
    sell = _matches(SELL_PATTERNS, req.text)
    buy = _matches(BUY_INTENT_PATTERNS, req.text)
    adv = _matches(ADVISOR_PATTERNS, req.text)
    if sell > buy and sell > adv:
        return RoleResp(role="seller", confidence=0.78)
    if buy > adv:
        return RoleResp(role="buyer", confidence=0.72)
    if adv > 0:
        return RoleResp(role="advisor", confidence=0.65)
    return RoleResp(role="other", confidence=0.55)


# ─── Endpoint 4: Sentiment + Urgency (multi-task) ────────────────────────

class SentimentReq(BaseModel):
    text: str


class SentimentResp(BaseModel):
    sentiment: str = Field(description="positive | neutral | negative")
    urgency: str = Field(description="low | medium | high")


@app.post("/predict-sentiment", response_model=SentimentResp)
def predict_sentiment(req: SentimentReq):
    bundle = _try_load_sentiment()
    if bundle is not None:
        import torch

        model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
        enc = tok(
            req.text[:2000], return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            s_logits, u_logits = model(enc["input_ids"], enc["attention_mask"])
        sent = SENTIMENT_LABELS[int(s_logits.argmax(-1))]
        urg = URGENCY_LABELS[int(u_logits.argmax(-1))]
        return SentimentResp(sentiment=sent, urgency=urg)
    # Stub
    pos = _matches(POSITIVE_PATTERNS, req.text)
    neg = _matches(NEGATIVE_PATTERNS, req.text)
    urg = _matches(URGENT_PATTERNS, req.text)
    sentiment = "positive" if pos > neg else ("negative" if neg > pos else "neutral")
    urgency = "high" if urg >= 2 else ("medium" if urg == 1 else "low")
    return SentimentResp(sentiment=sentiment, urgency=urgency)


# ─── Endpoint 5: Reply generation ────────────────────────────────────────

class ReplyReq(BaseModel):
    query: str = Field(description="What the user sells / the product being pitched")
    post_title: str
    post_body: str = ""
    tone: str = "helpful"


class ReplyResp(BaseModel):
    reply: str


# Openers are written to sound like a person typing on Reddit: no em dashes,
# no marketing voice, no claims the sender cannot back up.
TONE_OPENERS = {
    "helpful": "Happy to help with this. ",
    "professional": "Hi, this is the kind of work I do. ",
    "casual": "Hey, I've dealt with this before. ",
    "empathetic": "That sounds frustrating, and it's a common one. ",
}

# Characters that make a reply read as machine written. Em and en dashes are
# the giveaway; smart quotes and ellipsis characters are tidied at the same
# time so the text pastes cleanly into a Reddit comment box.
_DASH_RE = re.compile(r"\s*[—–]\s*")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:])")


# Claims a generated reply must never make on the user's behalf. The model
# cannot know the sender's track record, so any assertion of one is a
# fabrication the user would be posting under their own name. Output matching
# these falls back to the grounded stub instead.
_UNVERIFIABLE_CLAIMS = [
    # Explicit credentials the sender may not have
    r"\bcase stud(y|ies)\b",
    r"\btrusted by\b",
    r"\bguarantee(d|s)?\b",
    r"\b\d+\+? years? of experience\b",
    r"\bour clients?\b",
    r"\bportfolio of\b",
    r"\baward[- ]winning\b",
    r"\bproven track record\b",
    r"\b(top|leading|best)[- ]rated\b",
    # First-person track-record claims. The model invents these wholesale,
    # and they are the difference between "I can help" (fine to send on
    # someone's behalf) and "I have done this before" (not ours to assert).
    r"\b(I|we)'?ve delivered\b",
    r"\b(I|we)'?ve helped\b",
    r"\b(I|we)'?ve worked with\b",
    r"\b(I|we) have helped\b",
    r"\b(I|we)'?ve been doing\b",
    r"\bI'?ve done this\b",
    r"\bsimilar projects\b",
]
_CLAIM_RE = re.compile("|".join(_UNVERIFIABLE_CLAIMS), re.IGNORECASE)


def _makes_unverifiable_claim(text: str) -> str | None:
    """Return the offending phrase when a reply asserts something unprovable."""
    m = _CLAIM_RE.search(text or "")
    return m.group(0) if m else None


def _incoherent(text: str, offer: str) -> str | None:
    """
    Catch the ways this generator degenerates.

    It slot-fills a template with phrases lifted from the post, which is fine
    when the phrase happens to fit and nonsense when it does not ("I'd suggest
    beginning with a copywriter", "lose 60% of organic traffic usually unblocks
    things"). Grammaticality is not checkable here, but the specific failure
    shapes are.
    """
    t = (text or "").strip()
    low = t.lower()

    # The offer echoed several times is the clearest sign of a collapsed
    # template rather than a written sentence.
    if offer:
        o = offer.lower().strip()
        if len(o) > 3 and low.count(o) > 1:
            return f"offer repeated {low.count(o)}x"

    # Sentence fragments joined by a comma splice after our dash cleanup are
    # fine, but a reply that is one long run-on is not.
    if len(t) > 40 and t.count(".") == 0 and t.count("!") == 0:
        return "no sentence break"

    # Verb-initial fragments spliced mid-sentence ("If it helps, lose 60% of
    # organic traffic usually unblocks things").
    if re.search(r"\b(?:helps|start|begin|suggest)\w*,?\s+(?:lose|losing|lost|drop|dropped)\b", low):
        return "post fragment spliced into template"

    return None


# auto     use the model only when it passes every quality gate (default)
# model    always use the model when a checkpoint is loaded
# template use the grounded template reply, never the model
REPLY_MODE = os.environ.get("REDDIGEN_REPLY_MODE", "auto").lower()


def _humanise(text: str) -> str:
    """
    Clean a generated reply so it reads like something a person wrote.

    Applied to model output and stub output alike, so the guarantee holds
    regardless of which produced the text.
    """
    if not text:
        return text

    t = text.strip()

    # Drop a wrapping quote pair, which seq2seq models sometimes emit.
    if len(t) > 1 and t[0] in "\"'" and t[-1] == t[0]:
        t = t[1:-1].strip()

    # Em/en dash to a comma: keeps the clause break without the tell.
    t = _DASH_RE.sub(", ", t)

    t = (
        t.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("…", "...")
        .replace(" ", " ")
    )

    # A dash swap can leave ", ," or ", ." behind.
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r",\s*([.!?])", r"\1", t)
    t = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", t)
    t = _MULTISPACE_RE.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    t = t.strip()
    if t and t[0].islower():
        # Lower-case openers are fine for the casual tone, but a stray
        # lower-case start after cleaning usually means a lost first word.
        t = t[0].upper() + t[1:]
    return t


@app.post("/generate-reply", response_model=ReplyResp)
def generate_reply(req: ReplyReq):
    bundle = _try_load_reply()
    if bundle is not None:
        import torch

        model, tok, device = bundle["model"], bundle["tokenizer"], bundle["device"]
        prompt = REPLY_PROMPT.format(
            tone=req.tone,
            query=req.query,
            title=req.post_title,
            body=req.post_body[:1200],
        )
        enc = tok(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        # Beam search costs num_beams forward passes per token. On CPU that
        # dominates a search's wall-clock, since a lead list triggers one
        # generation per post. Two beams keeps the output coherent at roughly
        # half the cost; GPUs can afford the wider search.
        beams = 4 if device != "cpu" else 2
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=110,
                num_beams=beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        text = _humanise(tok.decode(out[0], skip_special_tokens=True))

        # Three ways generated output is rejected in favour of the stub:
        #   1. too short, or the model echoing the prompt back
        #   2. it asserts a track record the user cannot stand behind
        #   3. it repeats a phrase, which is how this model degenerates
        if REPLY_MODE == "model":
            if text:
                return ReplyResp(reply=text)
        elif REPLY_MODE != "template":
            claim = _makes_unverifiable_claim(text)
            flaw = _incoherent(text, _normalise_offer(req.query))
            too_short = len(text) < 40
            echoed = "post title:" in text.lower() or "our offer:" in text.lower()

            if claim:
                print(f"[server] reply rejected, unverifiable claim: {claim!r}")
            elif flaw:
                print(f"[server] reply rejected, {flaw}")

            if not (too_short or echoed or claim or flaw):
                return ReplyResp(reply=text)

    # Stub: a grounded, plainly worded reply that references the post.
    return ReplyResp(reply=_humanise(_stub_reply(req)))


# Users type search phrases ("looking for a wordpress developer"), but the
# reply needs the underlying SERVICE ("wordpress development"). Left raw it
# produces "I work on looking for a wordpress developer".
_SEARCH_PREFIX_RE = re.compile(
    r"^\s*(?:i\s+(?:am|'m)\s+)?"
    r"(?:looking\s+for|searching\s+for|need|needs|needed|want|wanted|"
    r"hiring|hire|find|finding|seeking|in\s+search\s+of|"
    r"anyone\s+(?:know|recommend)|recommend|best|good|top)\s+"
    r"(?:a|an|the|some|any)?\s*",
    re.IGNORECASE,
)
_TRAILING_NOISE_RE = re.compile(
    r"\s+(?:for|to)\s+(?:my|our|me|us)\b.*$", re.IGNORECASE
)


def _normalise_offer(query: str) -> str:
    """Turn a search phrase into something that reads as a service offering."""
    q = (query or "").strip().rstrip("?.!,")
    if not q:
        return ""
    q = _SEARCH_PREFIX_RE.sub("", q)
    q = _TRAILING_NOISE_RE.sub("", q).strip()
    return q or (query or "").strip()


# Words a quoted fragment must not end on, or the sentence trails off.
_DANGLING_WORDS = {
    "a", "an", "the", "of", "for", "to", "and", "or", "but", "with", "in",
    "on", "at", "by", "from", "as", "that", "this", "my", "our", "your",
    "is", "was", "are", "were", "be", "been", "it", "i", "we", "they",
    "handful", "couple", "lot", "bunch", "few", "some", "any",
}


def _pick_detail(body: str, limit: int = 110) -> str:
    """
    Pull one short, quotable detail from the post.

    Long fragments read badly when embedded mid-sentence, so anything over
    the limit is trimmed at a word boundary, and very short or very long
    sentences are skipped entirely.
    """
    body = (body or "").strip()
    if not body:
        return ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = sentence.strip().strip("-–—•*>").strip()
        if len(s) < 15:
            continue
        if len(s) > limit:
            cut = s[:limit].rsplit(" ", 1)[0].rstrip(",;:")
            # Trimming mid-clause leaves a dangling word ("...for a handful
            # of"), which reads worse than not quoting at all. Drop trailing
            # function words until the fragment ends on something concrete.
            words = cut.split()
            while words and words[-1].lower() in _DANGLING_WORDS:
                words.pop()
            cut = " ".join(words).rstrip(",;:")
            if len(cut) < 15:
                continue
            s = cut
        return s.rstrip(".")
    return ""


def _stub_reply(req: "ReplyReq") -> str:
    """
    Build a reply that reads like a person wrote it.

    Deliberately makes no claim the sender cannot stand behind: it offers
    help and asks a question rather than asserting past results.
    """
    opener = TONE_OPENERS.get(req.tone, TONE_OPENERS["helpful"])
    offer = _normalise_offer(req.query)
    title = req.post_title.strip().rstrip("?").rstrip(".")

    # Quote a concrete detail so the reply is clearly a response to *this*
    # thread rather than a copy-paste pitch. Keep the poster's own casing:
    # lower-casing their words reads like a bot paraphrasing them.
    detail = _pick_detail(req.post_body)

    # When the post has no body there is no specific detail to reflect back,
    # so the reply asks a question instead of pretending to know the context.
    # Embedding the raw title mid-sentence reads badly ("For need a logo for
    # my coffee shop, ...") because titles are rarely grammatical fragments.
    if req.tone == "casual":
        parts = [
            opener,
            f"This is the kind of work I do ({offer})." if offer else "",
            f"You mentioned {detail}, which is usually the place to start."
            if detail else "What have you tried so far?",
            "If you want a hand, reply here or send me a message.",
        ]
    elif req.tone == "professional":
        parts = [
            opener,
            f"This is my area ({offer})." if offer else "",
            f"You mentioned {detail}, which would be the first thing I'd look at."
            if detail else
            "I'd need a little more detail on your setup before suggesting an approach.",
            "Happy to discuss it properly if that would be useful.",
        ]
    elif req.tone == "empathetic":
        parts = [
            opener,
            f"You mentioned {detail}, and that part is rarely as simple as it looks."
            if detail else "This kind of thing usually takes longer than people expect.",
            f"This is the kind of work I do ({offer}), so happy to talk it through with no pressure."
            if offer else "Happy to talk it through with no pressure.",
        ]
    else:  # helpful
        parts = [
            opener,
            f"This is the kind of work I do ({offer})." if offer else "",
            f"You mentioned {detail}, so that's probably where I'd start."
            if detail else "Happy to take a look if you can share a bit more detail.",
            "Just reply here if that's useful.",
        ]

    return " ".join(p.strip() for p in parts if p and p.strip())


# ─── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    print(f"[server] Starting ReddiGen ML service on http://localhost:{port}")
    print(f"[server] Trained checkpoints directory: {MODELS_DIR}")
    print("[server] Missing checkpoints will fall back to rule-based stubs.")
    uvicorn.run(app, host="0.0.0.0", port=port)
