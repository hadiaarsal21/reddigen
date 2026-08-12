import Link from 'next/link';
import { DashboardShell } from '@/components/DashboardShell';
import { Icon } from '@/components/Icon';

export default function HomePage() {
  return (
    <DashboardShell activeHref="/" crumb="Overview">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <span className="pill brand">Local ML research build</span>
        <span className="pill neutral">v1.0</span>
      </div>

      <h1 className="page-title">
        Discover buying-intent on Reddit with{' '}
        <span style={{ color: 'var(--brand-primary)' }}>trained models</span>
      </h1>
      <p className="page-subtitle" style={{ maxWidth: 720 }}>
        Five deep-learning models — intent classification, semantic relevance ranking,
        role classification, multi-task sentiment + urgency, and reply generation —
        chained into one search pipeline. Everything runs locally on your machine.
      </p>

      <div style={{ display: 'flex', gap: 10, marginBottom: 32 }}>
        <Link href="/dashboard" className="btn btn-primary">
          <Icon name="search" size={14} /> Try the search
        </Link>
        <Link href="/models" className="btn btn-ghost">
          <Icon name="brain" size={14} /> View the ML architecture
        </Link>
      </div>

      <h2 style={{ fontSize: 18, margin: '32px 0 16px', letterSpacing: '-0.01em' }}>
        The pipeline
      </h2>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: 12,
        }}
      >
        {PIPELINE.map((step) => (
          <div className="card" key={step.title}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span className="pill brand">{step.tag}</span>
            </div>
            <h3>{step.title}</h3>
            <p>{step.body}</p>
          </div>
        ))}
      </div>

      <h2 style={{ fontSize: 18, margin: '40px 0 16px', letterSpacing: '-0.01em' }}>
        Features
      </h2>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 12,
        }}
      >
        <Link href="/dashboard" className="card" style={{ display: 'block' }}>
          <Icon name="search" size={18} style={{ color: 'var(--brand-primary)' }} />
          <h3 style={{ marginTop: 8 }}>Search</h3>
          <p>Type what you sell, run the full 5-model pipeline against fresh Reddit data.</p>
        </Link>
        <Link href="/deep-scan" className="card" style={{ display: 'block' }}>
          <Icon name="layers" size={18} style={{ color: 'var(--brand-primary)' }} />
          <h3 style={{ marginTop: 8 }}>Deep Scan</h3>
          <p>Mine comment threads for buyers hidden under other people&apos;s posts.</p>
        </Link>
        <Link href="/discover" className="card" style={{ display: 'block' }}>
          <Icon name="compass" size={18} style={{ color: 'var(--brand-primary)' }} />
          <h3 style={{ marginTop: 8 }}>Discover Subreddits</h3>
          <p>Semantic search ranks the subreddits most aligned with your offer.</p>
        </Link>
        <Link href="/leads" className="card" style={{ display: 'block' }}>
          <Icon name="list" size={18} style={{ color: 'var(--brand-primary)' }} />
          <h3 style={{ marginTop: 8 }}>Leads</h3>
          <p>Saved scored leads in a local SQLite database with reply drafts.</p>
        </Link>
      </div>

      <div className="card" style={{ marginTop: 32 }}>
        <h3>Getting started</h3>
        <p style={{ marginTop: 8, lineHeight: 1.7 }}>
          Open two terminals:
        </p>
        <div
          style={{
            marginTop: 8,
            padding: 14,
            background: 'var(--bg-surface-alt)',
            borderRadius: 8,
            fontFamily: 'JetBrains Mono, ui-monospace, monospace',
            fontSize: 12.5,
            color: 'var(--text-primary)',
            lineHeight: 1.9,
          }}
        >
          <div><span style={{ color: 'var(--text-tertiary)' }}>$</span> python ml/server.py <span style={{ color: 'var(--text-tertiary)' }}># ML server on :8000</span></div>
          <div><span style={{ color: 'var(--text-tertiary)' }}>$</span> npm run dev <span style={{ color: 'var(--text-tertiary)' }}># Next.js on :3000</span></div>
        </div>
      </div>
    </DashboardShell>
  );
}

const PIPELINE = [
  { tag: 'Step 1', title: 'Retrieve', body: 'Fetch Reddit posts via public JSON for the user query.' },
  { tag: 'Step 2', title: 'Classify intent', body: 'DistilBERT filters to buying-intent and advice-seeking posts.' },
  { tag: 'Step 3', title: 'Rank relevance', body: 'Sentence-BERT bi-encoder scores each post semantically.' },
  { tag: 'Step 4', title: 'Analyse', body: 'Multi-task RoBERTa predicts sentiment + urgency in one pass.' },
  { tag: 'Step 5', title: 'Attribute', body: 'Role classifier tags buyer / seller / advisor.' },
  { tag: 'Step 6', title: 'Generate', body: 'LoRA-tuned FLAN-T5 drafts a tone-conditioned reply.' },
];
