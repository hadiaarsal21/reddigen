import Link from 'next/link';

export default function HomePage() {
  return (
    <div>
      <h1 className="rg-h1">
        <span className="rg-accent">ReddiGen</span> — local ML research build
      </h1>
      <p className="rg-lede">
        Five trained deep-learning models discover, classify, and rank buying-intent
        conversations on Reddit. This is a self-contained local build — Next.js on
        port 3000, FastAPI model server on port 8000, SQLite for storage. Zero
        external API keys.
      </p>

      <div style={{ display: 'flex', gap: 12, marginBottom: 32 }}>
        <Link href="/dashboard" className="rg-btn">
          Try the search →
        </Link>
        <Link href="/models" className="rg-btn-ghost">
          View the ML architecture
        </Link>
      </div>

      <h2 className="rg-h2">The pipeline</h2>
      <div className="rg-grid-3">
        <div className="rg-card">
          <div className="rg-tag">Step 1</div>
          <h3 style={{ margin: '10px 0 6px', fontSize: 16 }}>Retrieve</h3>
          <p style={{ margin: 0, color: 'var(--rg-ink-dim)', fontSize: 14, lineHeight: 1.55 }}>
            Fetch Reddit posts via public JSON endpoints for the user&apos;s query.
          </p>
        </div>
        <div className="rg-card">
          <div className="rg-tag">Step 2</div>
          <h3 style={{ margin: '10px 0 6px', fontSize: 16 }}>Classify</h3>
          <p style={{ margin: 0, color: 'var(--rg-ink-dim)', fontSize: 14, lineHeight: 1.55 }}>
            Intent classifier (DistilBERT) filters to buying-intent posts only.
          </p>
        </div>
        <div className="rg-card">
          <div className="rg-tag">Step 3</div>
          <h3 style={{ margin: '10px 0 6px', fontSize: 16 }}>Rank</h3>
          <p style={{ margin: 0, color: 'var(--rg-ink-dim)', fontSize: 14, lineHeight: 1.55 }}>
            Sentence-BERT bi-encoder scores relevance to the query semantically.
          </p>
        </div>
        <div className="rg-card">
          <div className="rg-tag">Step 4</div>
          <h3 style={{ margin: '10px 0 6px', fontSize: 16 }}>Analyse</h3>
          <p style={{ margin: 0, color: 'var(--rg-ink-dim)', fontSize: 14, lineHeight: 1.55 }}>
            Multi-task RoBERTa predicts sentiment and urgency in one forward pass.
          </p>
        </div>
        <div className="rg-card">
          <div className="rg-tag">Step 5</div>
          <h3 style={{ margin: '10px 0 6px', fontSize: 16 }}>Attribute</h3>
          <p style={{ margin: 0, color: 'var(--rg-ink-dim)', fontSize: 14, lineHeight: 1.55 }}>
            Role classifier (buyer / seller / advisor) — filters out sellers pitching.
          </p>
        </div>
        <div className="rg-card">
          <div className="rg-tag">Step 6</div>
          <h3 style={{ margin: '10px 0 6px', fontSize: 16 }}>Generate</h3>
          <p style={{ margin: 0, color: 'var(--rg-ink-dim)', fontSize: 14, lineHeight: 1.55 }}>
            LoRA-tuned FLAN-T5 drafts a tone-conditioned reply for each lead.
          </p>
        </div>
      </div>

      <h2 className="rg-h2">Getting started</h2>
      <div className="rg-card">
        <p style={{ margin: 0, color: 'var(--rg-ink-dim)', fontSize: 14, lineHeight: 1.7 }}>
          Start both services with two terminals:<br />
          <code style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--rg-ink)', fontSize: 13 }}>
            $ npm run ml:serve
          </code>
          &nbsp;&nbsp;— starts the FastAPI ML server on <code style={{ color: 'var(--rg-accent)' }}>localhost:8000</code><br />
          <code style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--rg-ink)', fontSize: 13 }}>
            $ npm run dev
          </code>
          &nbsp;&nbsp;— starts the Next.js dashboard on <code style={{ color: 'var(--rg-accent)' }}>localhost:3000</code><br />
          <br />
          Then open <Link href="/dashboard" style={{ color: 'var(--rg-accent)' }}>/dashboard</Link> and run
          a search. See <Link href="/models" style={{ color: 'var(--rg-accent)' }}>/models</Link> for the
          full ML architecture and how to train each model.
        </p>
      </div>
    </div>
  );
}
