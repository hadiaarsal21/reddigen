'use client';

import { useState } from 'react';
import { DashboardShell } from '@/components/DashboardShell';
import { ChipSelect } from '@/components/ChipSelect';
import { SearchPanel } from '@/components/SearchPanel';
import { QuotaNote } from '@/components/QuotaNote';
import { CopyButton } from '@/components/CopyButton';
import { ResultNotice } from '@/components/ResultNotice';
import { Icon } from '@/components/Icon';
import { DEFAULT_POST_LIMIT } from '@/lib/limits';
import { buildLimitOptions, TIME_OPTIONS, TONE_OPTIONS } from '@/lib/options';

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

const LIMIT_OPTIONS = buildLimitOptions('leads');

interface SearchStats {
  requested: number;
  delivered: number;
  exact?: boolean;
  fetched: number;
  scored: number;
  rounds: number;
  qualified?: number;
  rejections?: Record<string, number>;
}

const EXAMPLES = [
  'looking for a python expert',
  'struggling with SEO',
  'need a copywriter',
  'best Notion template',
];

export default function DashboardPage() {
  const [query, setQuery] = useState('');
  const [tone, setTone] = useState('helpful');
  const [time, setTime] = useState('week');
  const [limit, setLimit] = useState(String(DEFAULT_POST_LIMIT));
  const [loading, setLoading] = useState(false);
  const [posts, setPosts] = useState<ScoredPost[]>([]);
  const [error, setError] = useState('');
  const [stats, setStats] = useState<SearchStats | null>(null);
  const [reason, setReason] = useState('');
  const [searched, setSearched] = useState(false);
  const [quota, setQuota] = useState<{ remaining: number; limit: number } | null>(null);

  async function handleSearch() {
    if (query.trim().length < 3) return;
    setLoading(true);
    setError('');
    setReason('');
    setSearched(true);
    setPosts([]);
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          tone,
          time,
          limit: Number(limit),
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || `Search failed (${res.status})`);
      }
      setPosts(data.posts || []);
      setStats(data.stats ?? null);
      setReason(data.reason ?? '');
      if (data.quota) setQuota(data.quota);
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

      <SearchPanel
        title="Find your next lead"
        subtitle="Describe what you're looking for — natural language works. The AI figures out the intent."
        placeholder="e.g. looking for a python expert…"
        value={query}
        onChange={setQuery}
        onSubmit={handleSearch}
        loading={loading}
        loadingLabel="Running pipeline…"
        submitLabel="Run Search"
        examples={EXAMPLES}
        filters={
          <>
            <ChipSelect
              icon="calendar"
              options={TIME_OPTIONS}
              value={time}
              onChange={setTime}
              ariaLabel="Time window"
            />
            <ChipSelect
              icon="target"
              options={LIMIT_OPTIONS}
              value={limit}
              onChange={setLimit}
              ariaLabel="Posts to scan"
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
            unit="searches"
            note="Asking for more leads searches Reddit from more angles, so it takes longer."
          />
        }
      />

      {error && (
        <ResultNotice kind="error" title="Search failed">
          {error}
        </ResultNotice>
      )}

      {!error && !loading && searched && posts.length === 0 && (
        <ResultNotice kind="empty" title="No matching leads found">
          {reason || 'Try different wording, or a longer time window.'}
        </ResultNotice>
      )}

      {/* Short of the requested number. Say why rather than padding the list
          with posts that do not match, which is the whole point of the gate. */}
      {!error && !loading && stats && posts.length > 0 && !stats.exact && (
        <ResultNotice
          kind="info"
          title={`Found ${stats.delivered} of the ${stats.requested} leads you asked for`}
        >
          Searched {stats.rounds} different way{stats.rounds === 1 ? '' : 's'} and
          checked {stats.scored} posts. Only {stats.delivered} showed real buying
          intent for this topic, so that is all you are seeing. The rest were
          filled with unrelated posts in earlier versions; now they are left out.
          Try broader wording or a longer time window for more.
        </ResultNotice>
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
        {p.reply && (
          <CopyButton text={p.reply} className="btn btn-primary" label="Copy reply" />
        )}
        <a className="btn btn-ghost" href={p.permalink} target="_blank" rel="noreferrer">
          Open on Reddit <Icon name="external" size={12} />
        </a>
        <button className="btn btn-ghost" onClick={() => onSave(p)} disabled={p.saved}>
          {p.saved ? (<><Icon name="check" size={13} /> Saved</>) : 'Save to leads'}
        </button>
      </div>
    </div>
  );
}
