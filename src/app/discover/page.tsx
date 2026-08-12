'use client';

import { useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { Icon } from '@/components/Icon';

interface DiscoveredSub {
  name: string;
  title: string;
  description: string;
  subscribers: number;
  active_users: number;
  mentions: number;
  sample_titles: string[];
  semantic_score: number;
  composite_score: number;
  url: string;
}

export default function DiscoverPage() {
  const [product, setProduct] = useState('');
  const [loading, setLoading] = useState(false);
  const [subs, setSubs] = useState<DiscoveredSub[]>([]);
  const [error, setError] = useState('');
  const [stats, setStats] = useState<{ posts_examined: number; subreddits_found: number } | null>(null);

  async function handleDiscover(e: React.FormEvent) {
    e.preventDefault();
    if (product.trim().length < 3) return;
    setLoading(true);
    setError('');
    setSubs([]);
    setStats(null);
    try {
      const res = await fetch('/api/discover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product: product.trim() }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `Discover failed (${res.status})`);
      }
      const data = await res.json();
      setSubs(data.subreddits || []);
      setStats(data.stats || null);
      if ((data.subreddits || []).length === 0) {
        setError('No relevant subreddits found. Try a more specific product description.');
      }
    } catch (err: any) {
      setError(err.message || 'Discover failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <DashboardShell activeHref="/discover" crumb="Discover Subreddits">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <h1 className="page-title" style={{ margin: 0 }}>Discover Subreddits</h1>
        <span className="pill purple">Semantic search</span>
      </div>
      <p className="page-subtitle">
        Finds subreddits most aligned with your offer by semantically ranking each
        candidate&apos;s public description against your product query, using the
        Sentence-BERT bi-encoder. Combines a semantic-similarity score with an
        activity-count boost so busy niche communities surface above dead ones.
      </p>

      <form onSubmit={handleDiscover} className="card" style={{ marginBottom: 24 }}>
        <label className="label">What do you sell?</label>
        <input
          type="text"
          className="input"
          placeholder="e.g. cold email deliverability tool"
          value={product}
          onChange={(e) => setProduct(e.target.value)}
          maxLength={200}
        />
        <div className="form-row" style={{ marginTop: 12 }}>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || product.trim().length < 3}
            style={{ marginLeft: 'auto' }}
          >
            {loading && <span className="spinner" />}
            {loading ? 'Discovering…' : 'Find subreddits'}
          </button>
        </div>
        <div className="hint" style={{ marginTop: 10 }}>
          Reddit&apos;s search API returns ~100 posts; we group by subreddit,
          fetch each community&apos;s description, embed with Sentence-BERT, and
          rank by cosine similarity + activity. Typically 20–40 seconds.
        </div>
      </form>

      {error && (
        <div className="card" style={{ borderColor: 'rgba(239,68,68,0.30)', background: 'var(--danger-bg)' }}>
          <p style={{ color: 'var(--danger)', margin: 0 }}>{error}</p>
        </div>
      )}

      {stats && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', gap: 24, fontSize: 13, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 600 }}>
                Posts examined
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, marginTop: 2 }}>{stats.posts_examined}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 600 }}>
                Subreddits ranked
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--brand-primary)', marginTop: 2 }}>
                {stats.subreddits_found}
              </div>
            </div>
          </div>
        </div>
      )}

      {subs.length > 0 && (
        <>
          <h2 style={{ fontSize: 16, margin: '24px 0 12px' }}>
            {subs.length} ranked {subs.length === 1 ? 'subreddit' : 'subreddits'}
          </h2>
          {subs.map((s) => <SubCard key={s.name} sub={s} />)}
        </>
      )}
    </DashboardShell>
  );
}

function SubCard({ sub }: { sub: DiscoveredSub }) {
  const scorePct = Math.round(sub.composite_score * 100);
  const scoreKind = scorePct >= 70 ? 'hot' : scorePct >= 40 ? 'warm' : 'cold';
  return (
    <div className="sub-card">
      <div>
        <a href={sub.url} target="_blank" rel="noreferrer" className="sub-name">
          r/{sub.name}
        </a>
        {sub.title && sub.title !== sub.name && (
          <span style={{ marginLeft: 8, color: 'var(--text-tertiary)', fontSize: 13 }}>
            — {sub.title}
          </span>
        )}
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <span className={`pill ${scoreKind}`}>score {scorePct}%</span>
      </div>
      <div className="sub-desc">
        {sub.description || <span style={{ color: 'var(--text-tertiary)' }}>No public description.</span>}
      </div>
      <div className="sub-metrics" style={{ gridColumn: '1 / -1' }}>
        <span><strong>{formatNum(sub.subscribers)}</strong> members</span>
        <span><strong>{formatNum(sub.active_users)}</strong> online</span>
        <span>mentioned in <strong>{sub.mentions}</strong> recent posts</span>
        <span>semantic <strong>{Math.round(sub.semantic_score * 100)}%</strong></span>
      </div>
      {sub.sample_titles.length > 0 && (
        <details style={{ gridColumn: '1 / -1', marginTop: 4 }}>
          <summary style={{ cursor: 'pointer', fontSize: 12, color: 'var(--text-tertiary)' }}>
            Sample matching posts ({sub.sample_titles.length})
          </summary>
          <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20, fontSize: 12.5, color: 'var(--text-secondary)' }}>
            {sub.sample_titles.map((t, i) => (
              <li key={i} style={{ marginBottom: 4 }}>{t}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function formatNum(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}
