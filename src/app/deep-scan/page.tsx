'use client';

import { useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { ChipSelect } from '@/components/ChipSelect';
import { SearchPanel } from '@/components/SearchPanel';
import { QuotaNote } from '@/components/QuotaNote';
import { CopyButton } from '@/components/CopyButton';
import { ResultNotice } from '@/components/ResultNotice';
import { Icon } from '@/components/Icon';
import { buildLimitOptions, TONE_OPTIONS } from '@/lib/options';

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
  /** 'comment' when mined from a thread, 'post' when the post itself is the lead. */
  source?: 'comment' | 'post';
}

interface Stats {
  posts_scanned: number;
  comments_examined: number;
  buyers_found: number;
  sellers_filtered: number;
  advisors_filtered: number;
  post_leads?: number;
  /** 'comments' | 'posts' | 'none' — where the returned leads came from. */
  source?: string;
  /** True when the buyer-attracting queries found nothing and we widened. */
  broadened?: boolean;
}

// Deep Scan is the heaviest feature — every post costs a comment-thread
// fetch — so it offers a shorter ladder than Search.
const SCAN_LIMIT_OPTIONS = buildLimitOptions('threads', 25);

const EXAMPLES = [
  'SEO agency for SaaS startups',
  'freelance video editor',
  'shopify store setup',
  'logo design service',
];

export default function DeepScanPage() {
  const [product, setProduct] = useState('');
  const [tone, setTone] = useState('helpful');
  const [limit, setLimit] = useState('10');
  const [loading, setLoading] = useState(false);
  const [leads, setLeads] = useState<CommentLead[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState('');
  const [quota, setQuota] = useState<{ remaining: number; limit: number } | null>(null);

  async function handleScan() {
    if (product.trim().length < 3) return;
    setLoading(true);
    setError('');
    setLeads([]);
    setStats(null);
    try {
      const res = await fetch('/api/deep-scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product: product.trim(), tone, limit: Number(limit) }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || `Scan failed (${res.status})`);
      }
      setLeads(data.leads || []);
      setStats(data.stats || null);
      if (data.quota) setQuota(data.quota);
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

      <SearchPanel
        title="Mine comment threads for buyers"
        subtitle="Describe what you sell. Deep Scan reads replies under other people's posts and keeps only the buyers."
        placeholder="e.g. SEO agency for SaaS startups…"
        value={product}
        onChange={setProduct}
        onSubmit={handleScan}
        loading={loading}
        loadingLabel="Mining comments…"
        submitLabel="Run Deep Scan"
        examples={EXAMPLES}
        filters={
          <>
            <ChipSelect
              icon="layers"
              options={SCAN_LIMIT_OPTIONS}
              value={limit}
              onChange={setLimit}
              ariaLabel="Threads to scan"
            />
            <ChipSelect
              icon="message"
              options={TONE_OPTIONS}
              value={tone}
              onChange={setTone}
              ariaLabel="Reply tone"
            />
            <ChipSelect
              icon="sliders"
              options={[]}
              value=""
              onChange={() => {}}
              label="Advanced"
              muted
              disabled
              ariaLabel="Advanced options (coming soon)"
            />
          </>
        }
        hint={
          <QuotaNote
            quota={quota}
            unit="deep scans"
            note="Each thread costs a comment fetch — typically 30–90 seconds."
          />
        }
      />

      {error && (
        <ResultNotice kind="error" title="Deep Scan failed">
          {error}
        </ResultNotice>
      )}

      {!error && !loading && stats && leads.length === 0 && (
        <ResultNotice kind="empty" title="No leads found in these threads">
          Reddit returned {stats.posts_scanned} post
          {stats.posts_scanned === 1 ? '' : 's'} and {stats.comments_examined} comment
          {stats.comments_examined === 1 ? '' : 's'}, and none of them showed buying
          intent. Try a broader product description, or widen the wording.
        </ResultNotice>
      )}

      {!error && leads.length > 0 && stats?.source === 'posts' && (
        <ResultNotice kind="info" title="Showing post authors, not commenters">
          The threads we found had no buyer comments, so Deep Scan fell back to the
          posts themselves and kept the authors showing buying intent. These are
          people asking publicly rather than replying under someone else&apos;s post.
        </ResultNotice>
      )}

      {!error && leads.length > 0 && stats?.broadened && (
        <ResultNotice kind="info" title="Search was widened">
          The hiring-style phrasings returned nothing, so the product term was
          searched directly over a longer time window.
        </ResultNotice>
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
            {lead.source === 'post' ? 'Post by the author:' : 'Comment under:'}
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
        {lead.reply && (
          <CopyButton text={lead.reply} className="btn btn-primary" label="Copy reply" />
        )}
        <a className="btn btn-ghost" href={lead.parent_post_url} target="_blank" rel="noreferrer">
          Open thread <Icon name="external" size={12} />
        </a>
        <button className="btn btn-ghost" onClick={() => onSave(lead)} disabled={lead.saved}>
          {lead.saved ? (<><Icon name="check" size={13} /> Saved</>) : 'Save to leads'}
        </button>
      </div>
    </div>
  );
}
