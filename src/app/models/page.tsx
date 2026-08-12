export default function ModelsPage() {
  return (
    <div>
      <h1 className="rg-h1">ML Models</h1>
      <p className="rg-lede">
        Five deep-learning models power the pipeline, each trained end-to-end
        on a labelled Reddit corpus. Every model is served from the local
        FastAPI process on <code>localhost:8000</code>. If a checkpoint is
        missing, the endpoint transparently falls back to a rule-based stub
        so the app never breaks.
      </p>

      <div className="rg-grid-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))' }}>
        <div className="rg-model-card">
          <h3>Intent classifier</h3>
          <div className="arch">DistilBERT · 4-class · cross-entropy</div>
          <p>
            Given a post title + body, predicts one of{' '}
            <code>buying_intent</code>, <code>advice_seeking</code>,{' '}
            <code>discussion</code>, or <code>off_topic</code>. Trained on
            ~10K labelled Reddit posts stratified by subreddit for genuine
            generalisation.
          </p>
          <p style={{ marginTop: 8, fontSize: 13 }}>
            <strong>Training:</strong> <code>python ml/train_intent.py</code>
          </p>
        </div>

        <div className="rg-model-card">
          <h3>Relevance ranker</h3>
          <div className="arch">Sentence-BERT · MiniLM · triplet loss</div>
          <p>
            Bi-encoder that embeds queries and posts into a shared 384-dim
            space where cosine similarity measures topical relevance. Trained
            with <code>MultipleNegativesRankingLoss</code> — no manual
            negative sampling needed.
          </p>
          <p style={{ marginTop: 8, fontSize: 13 }}>
            <strong>Training:</strong> <code>python ml/train_relevance.py</code>
          </p>
        </div>

        <div className="rg-model-card">
          <h3>Role classifier</h3>
          <div className="arch">RoBERTa · 3-class · focal loss</div>
          <p>
            Distinguishes <code>buyer</code> / <code>seller</code> / <code>advisor</code>{' '}
            on Reddit comments — the core of Deep Scan. Focal loss + class
            weights compensate for the built-in imbalance (buyers are rare).
          </p>
          <p style={{ marginTop: 8, fontSize: 13 }}>
            <strong>Training:</strong> <code>python ml/train_role.py</code>
          </p>
        </div>

        <div className="rg-model-card">
          <h3>Sentiment + urgency (multi-task)</h3>
          <div className="arch">RoBERTa · shared encoder · two heads</div>
          <p>
            One encoder, two prediction heads — sentiment (positive / neutral /
            negative) and urgency (low / medium / high) share features and are
            served in a single forward pass. Halves inference cost vs. two
            separate models.
          </p>
          <p style={{ marginTop: 8, fontSize: 13 }}>
            <strong>Training:</strong> <code>python ml/train_sentiment.py</code>
          </p>
        </div>

        <div className="rg-model-card">
          <h3>Reply generator</h3>
          <div className="arch">FLAN-T5-base · LoRA · seq2seq</div>
          <p>
            Fine-tuned with LoRA adapters — only ~0.5% of the parameters are
            updated, making training feasible on a single consumer GPU.
            Conditioned on tone (helpful / professional / casual / empathetic)
            and grounded in the specific post.
          </p>
          <p style={{ marginTop: 8, fontSize: 13 }}>
            <strong>Training:</strong> <code>python ml/train_reply.py</code>
          </p>
        </div>

        <div className="rg-model-card" style={{ borderColor: 'rgba(255,69,0,0.35)' }}>
          <h3>Full guide</h3>
          <div className="arch">→ MODELS-GUIDE.md</div>
          <p>
            The repo root contains <code>MODELS-GUIDE.md</code> — a detailed
            document with architectures, training data schemas, hyperparameters,
            evaluation metrics, deployment notes, and how each model plugs
            into the Next.js dashboard.
          </p>
        </div>
      </div>

      <h2 className="rg-h2">Pipeline flow</h2>
      <pre
        style={{
          background: 'var(--rg-bg-2)',
          border: '1px solid var(--rg-line)',
          borderRadius: 12,
          padding: 20,
          fontSize: 13,
          fontFamily: 'JetBrains Mono, monospace',
          color: 'var(--rg-ink-dim)',
          lineHeight: 1.7,
          overflowX: 'auto',
        }}
      >{`User query
    ↓
Reddit search (raw fetch)   →   N raw posts
    ↓
Intent classifier            →   posts with intent=buying_intent kept
    ↓
Relevance ranker             →   cosine similarity(query, post) as score
    ↓
Sentiment + urgency          →   two-head prediction per post
    ↓
Role classifier              →   role assigned (buyer/seller/advisor)
    ↓
Reply generator              →   tone-conditioned reply per surviving lead
    ↓
Ranked, scored, actionable leads shown in the dashboard
`}</pre>
    </div>
  );
}
