'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { DashboardShell } from '@/components/DashboardShell';
import { Icon } from '@/components/Icon';

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
  foundVia: string;
  createdAt: string;
}

const TABS = [
  { id: 'all', label: 'All' },
  { id: 'pending', label: 'Pending' },
  { id: 'replied', label: 'Replied' },
];

export default function LeadsPage() {
  const [leads, setLeads] = useState<Lead[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState('all');

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

  const filtered =
    tab === 'all' ? leads : leads.filter((l) => l.status === tab);

  return (
    <DashboardShell activeHref="/leads" crumb="Leads">
      <h1 className="page-title">Leads</h1>
      <p className="page-subtitle">
        Scored leads saved from Search, Deep Scan, and Discover. Storage is a local
        SQLite database — <code>prisma/dev.db</code>.
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid var(--border-subtle)' }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '10px 16px',
              fontSize: 13,
              fontWeight: 600,
              background: 'transparent',
              border: 'none',
              borderBottom: `2px solid ${tab === t.id ? 'var(--brand-primary)' : 'transparent'}`,
              color: tab === t.id ? 'var(--brand-primary)' : 'var(--text-secondary)',
              cursor: 'pointer',
              marginBottom: -1,
            }}
          >
            {t.label}{' '}
            <span style={{ color: 'var(--text-tertiary)', fontWeight: 500 }}>
              {t.id === 'all' ? leads.length : leads.filter((l) => l.status === t.id).length}
            </span>
          </button>
        ))}
      </div>

      {loading && <div className="empty"><span className="spinner" /> Loading…</div>}

      {!loading && filtered.length === 0 && (
        <div className="empty">
          <p>No leads yet.</p>
          <Link href="/dashboard" className="btn btn-primary" style={{ marginTop: 16 }}>
            <Icon name="search" size={14} /> Run your first search
          </Link>
        </div>
      )}

      {filtered.map((l) => {
        const relPct = Math.round(l.relevanceScore * 100);
        const relKind = relPct >= 70 ? 'hot' : relPct >= 40 ? 'warm' : 'cold';
        const urgKind = l.urgency === 'high' ? 'hot' : l.urgency === 'medium' ? 'warm' : 'cold';
        return (
          <div key={l.id} className="lead">
            <div className="lead-head">
              <div style={{ minWidth: 0, flex: 1 }}>
                <a href={l.url} target="_blank" rel="noreferrer" className="lead-title">
                  {l.title}
                </a>
                <div className="lead-meta">
                  r/{l.subreddit} · u/{l.author} · saved {new Date(l.createdAt).toLocaleString()} · via {l.foundVia}
                </div>
              </div>
              <span className={`pill ${l.status === 'replied' ? 'success' : 'neutral'}`}>
                {l.status}
              </span>
            </div>
            {l.selftext && (
              <div className="lead-body">
                {l.selftext.slice(0, 260)}{l.selftext.length > 260 ? '…' : ''}
              </div>
            )}
            <div className="lead-pills">
              <span className={`pill ${relKind}`}>relevance {relPct}%</span>
              {l.leadType && <span className="pill brand">intent: {l.leadType.replace('_', ' ')}</span>}
              {l.urgency && <span className={`pill ${urgKind}`}>urgency: {l.urgency}</span>}
              {l.role && <span className="pill purple">role: {l.role}</span>}
            </div>
            {l.suggestedReply && <div className="reply-box">{l.suggestedReply}</div>}
            <div className="lead-actions">
              {l.status !== 'replied' && (
                <button className="btn btn-ghost" onClick={() => toggleStatus(l.id, 'replied')}>
                  <Icon name="check" size={13} /> Mark replied
                </button>
              )}
              {l.status === 'replied' && (
                <button className="btn btn-ghost" onClick={() => toggleStatus(l.id, 'pending')}>
                  <Icon name="refresh" size={13} /> Un-mark
                </button>
              )}
              <button className="btn btn-ghost" onClick={() => del(l.id)}>
                <Icon name="trash" size={13} /> Delete
              </button>
              <a className="btn btn-ghost" href={l.url} target="_blank" rel="noreferrer">
                Open <Icon name="external" size={12} />
              </a>
            </div>
          </div>
        );
      })}
    </DashboardShell>
  );
}
