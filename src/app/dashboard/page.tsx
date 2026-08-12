'use client';

import { useState } from 'react';

interface ScoredPost {
  id: string;
  title: string;
  selftext: string;
  author: string;
  subreddit: string;
  permalink: string;
  score: number;
  num_comments: number;
  intent: string;
  intent_confidence: number;
  relevance: number;
  sentiment: string;
  urgency: string;
  role: string;
  reply: string;
  saved?: boolean;
}

const TONES = [
  { id: 'helpful', label: 'Helpful' },
  { id: 'professional', label: 'Professional' },
  { id: 'casual', label: 'Casual' },
  { id: 'empathetic', label: 'Empathetic' },
];

export default function DashboardPage() {
  const [query, setQuery] = useState('');
  const [tone, setTone] = useState('helpful');
  const [time, setTime] = useState('week');
  const [loading, setLoading] = useState(false);
  const [posts, setPosts] = useState<ScoredPost[]>([]);
  const [error, setError] = useState('');
  const [stage, setStage] = useState('');

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim().length < 3) return;
    setLoading(true);
    setError('');
    setPosts([]);
    setStage('Fetching Reddit posts…');
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), tone, time }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `Search failed (${res.status})`);
      }
      const data = await res.json();
      setPosts(data.posts || []);
      if ((data.posts || []).length === 0) {
        setError(
          'No high-intent leads matched. Try a different query, or widen the time window to "month".',
        );
      }
    } catch (err: any) {
      setError(err.message || 'Search failed');
    } finally {
      setLoading(false);
      setStage('');
    }
  }

  async function saveLead(p: ScoredPost) {
    try {
      const res = await fetch('/api/leads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          redditId: p.id,
          title: p.title,
          subreddit: p.subreddit,
          url: p.permalink,
          author: p.author,
          selftext: p.selftext,
          score: p.score,
          numComments: p.num_comments,
          keyword: query.trim(),
          relevanceScore: p.relevance,
          sentiment: p.sentiment,
          urgency: p.urgency,
          leadType: p.intent,
          role: p.role,
          suggestedReply: p.reply,
        }),
      });
      if (res.ok) {
        setPosts((prev) => prev.map((x) => (x.id === p.id ? { ...x, saved: true } : x)));
      }
    } catch {}
  }

  return (
    <div>
      <h1 className="rg-h1">Dashboard</h1>
      <p className="rg-lede">
        Type what you sell. The pipeline retrieves Reddit posts, classifies intent,
        ranks relevance, analyses sentiment + urgency + role, and drafts a reply —
        all locally on your machine.
      </p>

      <form onSubmit={handleSearch} className="rg-card" style={{ marginBottom: 32 }}>
        <input
          type="text"
          className="rg-input"
          placeholder="e.g. python freelancer for automation"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          maxLength={200}
          style={{ marginBottom: 12 }}
        />
        <div className="rg-row">
          <label style={{ fontSize: 12, color: 'var(--rg-ink-mute)' }}>
            Reply tone{' '}
            <select
              className="rg-select"
              value={tone}
              onChange={(e) => setTone(e.target.value)}
              style={{ marginLeft: 4 }}
            >
              {TONES.map((t) => (
                <option key={t.id} value={t.id}>{t.label}</option>
              ))}
            </select>
          </label>
          <label style={{ fontSize: 12, color: 'var(--rg-ink-mute)' }}>
            Time window{' '}
            <select
              className="rg-select"
              value={time}
              onChange={(e) => setTime(e.target.value)}
              style={{ marginLeft: 4 }}
            >
              <option value="day">Past day</option>
              <option value="week">Past week</option>
              <option value="month">Past month</option>
              <option value="year">Past year</option>
            </select>
          </label>
          <button
            type="submit"
            className="rg-btn"
            disabled={loading || query.trim().length < 3}
            style={{ marginLeft: 'auto' }}
          >
            {loading && <span className="rg-loading" />}
            {loading ? 'Running pipeline…' : 'Run search'}
          </button>
        </div>
        {stage && (
          <div style={{ marginTop: 12, fontSize: 12, color: 'var(--rg-ink-mute)', fontFamily: 'JetBrains Mono, monospace' }}>
            {stage}
          </div>
        )}
      </form>

      {error && (
        <div className="rg-card" style={{ borderColor: 'rgba(239,68,68,0.35)', background: 'rgba(239,68,68,0.06)' }}>
          <p style={{ margin: 0, color: 'var(--rg-bad)' }}>{error}</p>
        </div>
      )}

      {posts.length > 0 && (
        <>
          <h2 className="rg-h2">{posts.length} scored {posts.length === 1 ? 'lead' : 'leads'}</h2>
          {posts.map((p) => <LeadCard key={p.id} p={p} onSave={saveLead} />)}
        </>
      )}
    </div>
  );
}

function LeadCard({ p, onSave }: { p: ScoredPost; onSave: (p: ScoredPost) => void }) {
  const relPct = Math.round(p.relevance * 100);
  const relTone: string = relPct >= 70 ? 'g' : relPct >= 40 ? 'w' : 'r';
  const urgTone: string = p.urgency === 'high' ? 'r' : p.urgency === 'medium' ? 'w' : '';
  const sentTone: string = p.sentiment === 'positive' ? 'g' : p.sentiment === 'negative' ? 'r' : '';
  return (
    <div className="rg-lead">
      <div className="rg-lead-head">
        <div>
          <a href={p.permalink} target="_blank" rel="noreferrer" className="rg-lead-title">
            {p.title}
          </a>
          <div className="rg-lead-meta">
            r/{p.subreddit} · u/{p.author} · {p.score} pts · {p.num_comments} comments
          </div>
        </div>
      </div>
      {p.selftext && (
        <div className="rg-lead-body">
          {p.selftext.slice(0, 300)}
          {p.selftext.length > 300 ? '…' : ''}
        </div>
      )}
      <div className="rg-lead-scores">
        <span className={`rg-score-pill rg-tag ${relTone}`}>relevance {relPct}%</span>
        <span className="rg-score-pill rg-tag">intent: {p.intent}</span>
        {p.sentiment && (
          <span className={`rg-score-pill rg-tag ${sentTone}`}>sentiment: {p.sentiment}</span>
        )}
        {p.urgency && (
          <span className={`rg-score-pill rg-tag ${urgTone}`}>urgency: {p.urgency}</span>
        )}
        <span className="rg-score-pill rg-tag">role: {p.role}</span>
      </div>
      {p.reply && (
        <>
          <div style={{ fontSize: 12, color: 'var(--rg-ink-mute)', margin: '4px 0 6px' }}>
            Draft reply
          </div>
          <div className="rg-reply">{p.reply}</div>
        </>
      )}
      <div className="rg-lead-actions" style={{ marginTop: 12 }}>
        <button className="rg-btn-ghost" onClick={() => onSave(p)} disabled={p.saved}>
          {p.saved ? '✓ Saved' : 'Save to leads'}
        </button>
        <a className="rg-btn-ghost" href={p.permalink} target="_blank" rel="noreferrer">
          Open on Reddit ↗
        </a>
      </div>
    </div>
  );
}
