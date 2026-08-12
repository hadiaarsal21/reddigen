'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

interface Lead {
  id: number;
  redditId: string;
  title: string;
  subreddit: string;
  url: string;
  author: string;
  selftext: string;
  relevanceScore: number;
  sentiment: string;
  urgency: string;
  leadType: string;
  role: string;
  suggestedReply: string;
  keyword: string;
  status: string;
  createdAt: string;
}

export default function LeadsPage() {
  const [leads, setLeads] = useState<Lead[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/leads')
      .then((r) => r.json())
      .then((d) => {
        setLeads(d.leads || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  async function toggleStatus(id: number, next: string) {
    setLeads((prev) => prev.map((l) => (l.id === id ? { ...l, status: next } : l)));
    await fetch('/api/leads', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, status: next }),
    });
  }

  async function del(id: number) {
    setLeads((prev) => prev.filter((l) => l.id !== id));
    await fetch(`/api/leads?id=${id}`, { method: 'DELETE' });
  }

  return (
    <div>
      <h1 className="rg-h1">Leads</h1>
      <p className="rg-lede">
        All the scored leads you&apos;ve saved from the dashboard. Storage is a
        local SQLite database at <code>prisma/dev.db</code>.
      </p>

      {loading && <div className="rg-empty"><span className="rg-loading" /> Loading…</div>}

      {!loading && leads.length === 0 && (
        <div className="rg-empty">
          <p>No leads saved yet.</p>
          <Link href="/dashboard" className="rg-btn" style={{ marginTop: 12 }}>
            Run your first search →
          </Link>
        </div>
      )}

      {leads.map((l) => {
        const relPct = Math.round(l.relevanceScore * 100);
        const relTone = relPct >= 70 ? 'g' : relPct >= 40 ? 'w' : 'r';
        const urgTone = l.urgency === 'high' ? 'r' : l.urgency === 'medium' ? 'w' : '';
        return (
          <div key={l.id} className="rg-lead">
            <div className="rg-lead-head">
              <div>
                <a href={l.url} target="_blank" rel="noreferrer" className="rg-lead-title">
                  {l.title}
                </a>
                <div className="rg-lead-meta">
                  r/{l.subreddit} · u/{l.author} · saved {new Date(l.createdAt).toLocaleString()}
                </div>
              </div>
              <span className={`rg-score-pill rg-tag ${l.status === 'replied' ? 'g' : ''}`}>
                {l.status}
              </span>
            </div>
            {l.selftext && (
              <div className="rg-lead-body">
                {l.selftext.slice(0, 240)}{l.selftext.length > 240 ? '…' : ''}
              </div>
            )}
            <div className="rg-lead-scores">
              <span className={`rg-score-pill rg-tag ${relTone}`}>relevance {relPct}%</span>
              {l.leadType && <span className="rg-score-pill rg-tag">intent: {l.leadType}</span>}
              {l.urgency && (
                <span className={`rg-score-pill rg-tag ${urgTone}`}>urgency: {l.urgency}</span>
              )}
              {l.role && <span className="rg-score-pill rg-tag">role: {l.role}</span>}
            </div>
            {l.suggestedReply && <div className="rg-reply">{l.suggestedReply}</div>}
            <div className="rg-lead-actions" style={{ marginTop: 12 }}>
              {l.status !== 'replied' && (
                <button className="rg-btn-ghost" onClick={() => toggleStatus(l.id, 'replied')}>
                  Mark replied
                </button>
              )}
              {l.status === 'replied' && (
                <button className="rg-btn-ghost" onClick={() => toggleStatus(l.id, 'pending')}>
                  Un-mark
                </button>
              )}
              <button className="rg-btn-ghost" onClick={() => del(l.id)}>Delete</button>
              <a className="rg-btn-ghost" href={l.url} target="_blank" rel="noreferrer">
                Open ↗
              </a>
            </div>
          </div>
        );
      })}
    </div>
  );
}
