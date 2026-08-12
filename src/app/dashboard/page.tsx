'use client';

import { useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { Icon } from '@/components/Icon';

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

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim().length < 3) return;
    setLoading(true);
    setError('');
    setPosts([]);
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
        setError('No high-intent leads matched. Try a different query, or widen the time window.');
      }
    } catch (err: any) {
      setError(err.message || 'Search failed');
    } finally {
      setLoading(false);
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
    <DashboardShell activeHref="/dashboard" crumb="Search">
      <h1 className="page-title">Search</h1>
      <p className="page-subtitle">
        Type what you sell. The pipeline retrieves Reddit posts and runs them through
        five trained models — intent, relevance, role, sentiment + urgency, and reply
        generation — locally on your machine.
      </p>

      <form onSubmit={handleSearch} className="card" style={{ marginBottom: 24 }}>
        <label className="label">What are you looking for?</label>
        <input
          type="text"
          className="input"
          placeholder="e.g. python freelancer for automation"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          maxLength={200}
        />
        <div className="form-row" style={{ marginTop: 12 }}>
          <div className="field">
            <label className="label">Reply tone</label>
            <select className="select" value={tone} onChange={(e) => setTone(e.target.value)}>
              {TONES.map((t) => (
                <option key={t.id} value={t.id}>{t.label}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label className="label">Time window</label>
            <select className="select" value={time} onChange={(e) => setTime(e.target.value)}>
              <option value="day">Past day</option>
              <option value="week">Past week</option>
              <option value="month">Past month</option>
              <option value="year">Past year</option>
            </select>
          </div>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || query.trim().length < 3}
            style={{ marginLeft: 'auto' }}
          >
            {loading && <span className="spinner" />}
            {loading ? 'Running pipeline…' : 'Run search'}
          </button>
        </div>
      </form>

      {error && (
        <div
          className="card"
          style={{ borderColor: 'rgba(239,68,68,0.30)', background: 'var(--danger-bg)' }}
        >
          <p style={{ color: 'var(--danger)', margin: 0 }}>{error}</p>
        </div>
      )}

      {posts.length > 0 && (
        <>
          <h2 style={{ fontSize: 16, margin: '24px 0 12px' }}>
            {posts.length} scored {posts.length === 1 ? 'lead' : 'leads'}
          </h2>
          {posts.map((p) => (
            <LeadCard key={p.id} p={p} onSave={saveLead} />
          ))}
        </>
      )}
    </DashboardShell>
  );
}

function LeadCard({ p, onSave }: { p: ScoredPost; onSave: (p: ScoredPost) => void }) {
  const relPct = Math.round(p.relevance * 100);
  const relKind = relPct >= 70 ? 'hot' : relPct >= 40 ? 'warm' : 'cold';
  const urgKind = p.urgency === 'high' ? 'hot' : p.urgency === 'medium' ? 'warm' : 'cold';
  const sentKind = p.sentiment === 'positive' ? 'success' : p.sentiment === 'negative' ? 'danger' : 'neutral';
  return (
    <div className="lead">
      <div className="lead-head">
        <div style={{ minWidth: 0, flex: 1 }}>
          <a href={p.permalink} target="_blank" rel="noreferrer" className="lead-title">
            {p.title}
          </a>
          <div className="lead-meta">
            r/{p.subreddit} · u/{p.author} · {p.score} pts · {p.num_comments} comments
          </div>
        </div>
      </div>
      {p.selftext && (
        <div className="lead-body">
          {p.selftext.slice(0, 300)}{p.selftext.length > 300 ? '…' : ''}
        </div>
      )}
      <div className="lead-pills">
        <span className={`pill ${relKind}`}>relevance {relPct}%</span>
        <span className="pill brand">intent: {p.intent.replace('_', ' ')}</span>
        {p.sentiment && <span className={`pill ${sentKind}`}>sentiment: {p.sentiment}</span>}
        {p.urgency && <span className={`pill ${urgKind}`}>urgency: {p.urgency}</span>}
        <span className="pill purple">role: {p.role}</span>
      </div>
      {p.reply && <div className="reply-box">{p.reply}</div>}
      <div className="lead-actions">
        <button className="btn btn-ghost" onClick={() => onSave(p)} disabled={p.saved}>
          {p.saved ? (<><Icon name="check" size={13} /> Saved</>) : 'Save to leads'}
        </button>
        <a className="btn btn-ghost" href={p.permalink} target="_blank" rel="noreferrer">
          Open on Reddit <Icon name="external" size={12} />
        </a>
      </div>
    </div>
  );
}
