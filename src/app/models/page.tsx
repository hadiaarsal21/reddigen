import { DashboardShell } from '@/components/DashboardShell';
import { Icon } from '@/components/Icon';

const MODELS = [
  {
    tag: 'Model 1',
    title: 'Intent classifier',
    arch: 'DistilBERT-base · 4-class · cross-entropy',
    body: 'Given a post title + body, predicts one of buying_intent, advice_seeking, discussion, or off_topic. First-stage filter that cuts the candidate pool ~90% before more expensive models run.',
    train: 'python ml/train_intent.py',
  },
  {
    tag: 'Model 2',
    title: 'Relevance ranker',
    arch: 'Sentence-BERT (MiniLM) · triplet loss',
    body: 'Bi-encoder mapping queries and posts into a shared 384-dim space where cosine similarity measures topical fit. Trained with MultipleNegativesRankingLoss — no manual negative sampling.',
    train: 'python ml/train_relevance.py',
  },
  {
    tag: 'Model 3',
    title: 'Role classifier',
    arch: 'RoBERTa-base · focal loss + class weights',
    body: 'Distinguishes buyer / seller / advisor / other on Reddit comments. The core of Deep Scan. Focal loss + inverse-frequency weights compensate for the built-in class imbalance (buyers are the rare and valuable class).',
    train: 'python ml/train_role.py',
  },
  {
    tag: 'Model 4',
    title: 'Sentiment + urgency (multi-task)',
    arch: 'RoBERTa-base · shared encoder · two heads',
    body: 'One encoder feeds two independent classification heads — sentiment (positive/neutral/negative) and urgency (low/medium/high). Halves inference cost vs. two separate models and empirically improves both tasks via feature sharing.',
    train: 'python ml/train_sentiment.py',
  },
  {
    tag: 'Model 5',
    title: 'Reply generator',
    arch: 'FLAN-T5-base · LoRA · seq2seq',
    body: 'Fine-tuned with LoRA adapters — only ~0.4% of parameters are updated. Trainable on a single 12–16 GB consumer GPU. Conditioned on tone (helpful / professional / casual / empathetic) and grounded in the specific post.',
    train: 'python ml/train_reply.py',
  },
];

export default function ModelsPage() {
  return (
    <DashboardShell activeHref="/models" crumb="ML Models">
      <h1 className="page-title">ML Models</h1>
      <p className="page-subtitle">
        Five deep-learning models power the pipeline. Every model is served from a
        single FastAPI process on <code>localhost:8000</code>. If a checkpoint isn&apos;t
        present, the endpoint falls back to a rule-based stub so the app always
        works end-to-end. Once training completes, the FastAPI server picks up
        the new checkpoint automatically on the next request — no restart needed.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        {MODELS.map((m) => (
          <div key={m.title} className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span className="pill brand">{m.tag}</span>
              <Icon name="brain" size={16} style={{ color: 'var(--brand-primary)' }} />
            </div>
            <h3>{m.title}</h3>
            <div
              style={{
                fontSize: 12,
                color: 'var(--brand-primary)',
                fontFamily: 'JetBrains Mono, monospace',
                margin: '4px 0 10px',
              }}
            >
              {m.arch}
            </div>
            <p>{m.body}</p>
            <div
              style={{
                marginTop: 12,
                padding: '8px 10px',
                background: 'var(--bg-surface-alt)',
                borderRadius: 6,
                fontSize: 12,
                fontFamily: 'JetBrains Mono, monospace',
                color: 'var(--text-primary)',
              }}
            >
              <span style={{ color: 'var(--text-tertiary)' }}>$</span> {m.train}
            </div>
          </div>
        ))}
      </div>

      <h2 style={{ fontSize: 18, margin: '32px 0 16px', letterSpacing: '-0.01em' }}>
        Pipeline flow (Search endpoint)
      </h2>
      <div className="card">
        <pre
          style={{
            margin: 0,
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 12.5,
            color: 'var(--text-secondary)',
            lineHeight: 1.85,
            whiteSpace: 'pre',
            overflowX: 'auto',
          }}
        >{`user query
    │
    ▼
Reddit search (public JSON)      →  N raw posts
    │
    ▼
[Model 1] intent classifier      →  drop off_topic + discussion (~-90%)
    │
    ▼
[Model 2] relevance ranker       →  cosine(query, post); drop below 0.20
    │
    ▼
[Model 4] sentiment + urgency    →  tag each surviving lead (multi-task)
    │
    ▼
[Model 3] role classifier        →  attribute buyer/seller/advisor
    │
    ▼
[Model 5] reply generator        →  tone-conditioned draft reply
    │
    ▼
scored, ranked, reply-drafted leads → dashboard`}</pre>
      </div>

      <h2 style={{ fontSize: 18, margin: '32px 0 16px', letterSpacing: '-0.01em' }}>
        Feature ↔ model mapping
      </h2>
      <div className="card">
        <table
          style={{
            width: '100%',
            fontSize: 13,
            borderCollapse: 'collapse',
          }}
        >
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-subtle)', textAlign: 'left' }}>
              <th style={{ padding: '10px 12px', fontWeight: 600, color: 'var(--text-tertiary)' }}>Feature</th>
              <th style={{ padding: '10px 12px', fontWeight: 600, color: 'var(--text-tertiary)' }}>Models used</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <td style={{ padding: '10px 12px', fontWeight: 600 }}>Search</td>
              <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>Intent, Relevance, Sentiment+Urgency, Role, Reply</td>
            </tr>
            <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <td style={{ padding: '10px 12px', fontWeight: 600 }}>Deep Scan</td>
              <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>Role (primary), Relevance, Sentiment+Urgency, Reply</td>
            </tr>
            <tr>
              <td style={{ padding: '10px 12px', fontWeight: 600 }}>Discover Subreddits</td>
              <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>Relevance (semantic subreddit ranking)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p style={{ marginTop: 24, color: 'var(--text-tertiary)', fontSize: 13 }}>
        Full architectures, training procedures, hyperparameters, and evaluation
        methodology are in{' '}
        <code style={{ color: 'var(--brand-primary)' }}>MODELS-GUIDE.md</code> at
        the repo root.
      </p>
    </DashboardShell>
  );
}
