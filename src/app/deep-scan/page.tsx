'use client';

import { useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { Icon } from '@/components/Icon';

interface CommentLead {
  id: string;
  commentId: string;
  author: string;
  body: string;
  subreddit: string;
  parent_post_id: string;
  parent_post_title: string;
  parent_post_url: string;
  role: string;
  role_confidence: number;
  relevance: number;
  sentiment: string;
  urgency: string;
  reply: string;
  saved?: boolean;
}

interface Stats {
  posts_scanned: number;
  comments_examined: number;
  buyers_found: number;
  sellers_filtered: number;
  advisors_filtered: number;
}

const TONES = [
  { id: 'helpful', label: 'Helpful' },
  { id: 'professional', label: 'Professional' },
  { id: 'casual', label: 'Casual' },
  { id: 'empathetic', label: 'Empathetic' },
];

export default function DeepScanPage() {
  const [product, setProduct] = useState('');
  const [tone, setTone] = useState('helpful');
  const [loading, setLoading] = useState(false);
  const [leads, setLeads] = useState<CommentLead[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState('');

  async function handleScan(e: React.FormEvent) {
    e.preventDefault();
    if (product.trim().length < 3) return;
    setLoading(true);
    setError('');
    setLeads([]);
    setStats(null);
    try {
      const res = await fetch('/api/deep-scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product: product.trim(), tone }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `Scan failed (${res.status})`);
      }
      const data = await res.json();
      setLeads(data.leads || []);
      setStats(data.stats || null);
      if ((data.leads || []).length === 0) {
        setError(
          'No buyer comments found. Try a broader product description, or wait — buyer replies accumulate over time.',
        );
      }
    } catch (err: any) {
      setError(err.message || 'Scan failed');
    } finally {
      setLoading(false);
    }
  }

  async function saveLead(l: CommentLead) {
    try {
      const res = await fetch('/api/leads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          redditId: l.id,
          title: `Comment on: ${l.parent_post_title}`,
          subreddit: l.subreddit,
          url: l.parent_post_url,
          author: l.author,
          selftext: l.body,
          keyword: product.trim(),
          relevanceScore: l.relevance,
          sentiment: l.sentiment,
          urgency: l.urgency,
          leadType: 'buying_intent',
          role: l.role,
          suggestedReply: l.reply,
          foundVia: 'deep_scan',
          parentPostId: l.parent_post_id,
          parentPostUrl: l.parent_post_url,
        }),
      });
      if (res.ok) {
        setLeads((prev) => prev.map((x) => (x.id === l.id ? { ...x, saved: true } : x)));
      }
    } catch {}
  }

  return (
    <DashboardShell activeHref="/deep-scan" crumb="Deep Scan">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <h1 className="page-title" style={{ margin: 0 }}>Deep Scan</h1>
        <span className="pill purple">Comment mining</span>
      </div>
      <p className="page-subtitle">
        Standard search only looks at post titles. Deep Scan reads the{' '}
        <strong>comment threads UNDER other people&apos;s posts</strong> to find buyers
        replying with &quot;I need this too!&quot; The role classifier (RoBERTa + focal loss)
        distinguishes buyers from sellers pitching and advisors dispensing opinions.
      </p>

      <form onSubmit={handleScan} className="card" style={{ marginBottom: 24 }}>
        <label className="label">Describe what you sell</label>
        <input
          type="text"
          className="input"
          placeholder="e.g. SEO agency for SaaS startups"
          value={product}
          onChange={(e) => setProduct(e.target.value)}
          maxLength={200}
        />
        <div className="form-row" style={{ marginTop: 12 }}>
          <div className="field" style={{ maxWidth: 200 }}>
            <label className="label">Reply tone</label>
            <select className="select" value={tone} onChange={(e) => setTone(e.target.value)}>
              {TONES.map((t) => (
                <option key={t.id} value={t.id}>{t.label}</option>
              ))}
            </select>
          </div>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || product.trim().length < 3}
            style={{ marginLeft: 'auto' }}
          >
            {loading && <span className="spinner" />}
            {loading ? 'Mining comments…' : 'Run Deep Scan'}
          </button>
        </div>
        <div className="hint" style={{ marginTop: 10 }}>
          Deep Scan searches for hiring / recommendation posts, fetches their
          comment threads, and role-classifies every reply. Typically finishes
          in 30–90 seconds.
        </div>
      </form>

      {error && (
        <div className="card" style={{ borderColor: 'rgba(239,68,68,0.30)', background: 'var(--danger-bg)' }}>
          <p style={{ color: 'var(--danger)', margin: 0 }}>{error}</p>
        </div>
      )}

      {stats && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', fontSize: 13 }}>
            <StatItem label="Posts scanned" value={stats.posts_scanned} />
            <StatItem label="Comments examined" value={stats.comments_examined} />
            <StatItem label="Buyers found" value={stats.buyers_found} highlight />
            <StatItem label="Sellers filtered" value={stats.sellers_filtered} />
            <StatItem label="Advisors filtered" value={stats.advisors_filtered} />
          </div>
        </div>
      )}

      {leads.length > 0 && (
        <>
          <h2 style={{ fontSize: 16, margin: '24px 0 12px' }}>
            {leads.length} buyer {leads.length === 1 ? 'comment' : 'comments'}
          </h2>
          {leads.map((l) => <CommentLeadCard key={l.id} lead={l} onSave={saveLead} />)}
        </>
      )}
    </DashboardShell>
  );
}

function StatItem({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 600 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: highlight ? 'var(--brand-primary)' : 'var(--text-primary)', marginTop: 2 }}>
        {value}
      </div>
    </div>
  );
}

function CommentLeadCard({ lead, onSave }: { lead: CommentLead; onSave: (l: CommentLead) => void }) {
  const relPct = Math.round(lead.relevance * 100);
  const relKind = relPct >= 70 ? 'hot' : relPct >= 40 ? 'warm' : 'cold';
  const urgKind = lead.urgency === 'high' ? 'hot' : lead.urgency === 'medium' ? 'warm' : 'cold';
  const sentKind = lead.sentiment === 'positive' ? 'success' : lead.sentiment === 'negative' ? 'danger' : 'neutral';
  return (
    <div className="lead">
      <div className="lead-head">
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginBottom: 6 }}>
            Comment under:
          </div>
          <a href={lead.parent_post_url} target="_blank" rel="noreferrer" className="lead-title">
            {lead.parent_post_title}
          </a>
          <div className="lead-meta">
            r/{lead.subreddit} · u/{lead.author}
          </div>
        </div>
      </div>
      <div className="lead-body" style={{ borderLeft: '3px solid var(--brand-primary)', paddingLeft: 12 }}>
        {lead.body}
      </div>
      <div className="lead-pills">
        <span className={`pill ${relKind}`}>relevance {relPct}%</span>
        <span className="pill purple">
          role: buyer ({Math.round(lead.role_confidence * 100)}%)
        </span>
        {lead.sentiment && <span className={`pill ${sentKind}`}>sentiment: {lead.sentiment}</span>}
        {lead.urgency && <span className={`pill ${urgKind}`}>urgency: {lead.urgency}</span>}
      </div>
      {lead.reply && <div className="reply-box">{lead.reply}</div>}
      <div className="lead-actions">
        <button className="btn btn-ghost" onClick={() => onSave(lead)} disabled={lead.saved}>
          {lead.saved ? (<><Icon name="check" size={13} /> Saved</>) : 'Save to leads'}
        </button>
        <a className="btn btn-ghost" href={lead.parent_post_url} target="_blank" rel="noreferrer">
          Open thread <Icon name="external" size={12} />
        </a>
      </div>
    </div>
  );
}
