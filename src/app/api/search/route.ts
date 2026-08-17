// POST /api/search
//
// The single endpoint the dashboard hits. Fetches Reddit posts for the
// query, then chains all five trained models to produce a scored,
// ranked, reply-drafted lead list.

import { NextRequest, NextResponse } from 'next/server';
import { searchReddit } from '@/lib/reddit';
import {
  clampPostLimit,
  clientKey,
  rateLimit,
  RATE_RULES,
  validateQuery,
} from '@/lib/limits';
import {
  classifyIntent,
  scoreRelevance,
  predictSentiment,
  classifyRole,
  generateReply,
} from '@/lib/mlClient';

export const dynamic = 'force-dynamic';

// Only posts with relevance >= this threshold survive the ranker.
const RELEVANCE_THRESHOLD = 0.20;

export async function POST(req: NextRequest) {
  // ── Rate limit ────────────────────────────────────────────────────────
  const rl = rateLimit(clientKey(req, 'search'), RATE_RULES.search);
  if (!rl.ok) {
    return NextResponse.json(
      {
        error: `Rate limit reached — ${rl.limit} searches per minute. Try again in ${rl.retryAfter}s.`,
        retryAfter: rl.retryAfter,
      },
      { status: 429, headers: { 'Retry-After': String(rl.retryAfter) } },
    );
  }

  let body: any = {};
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 });
  }

  const query: string = (body.query || '').toString().trim();
  const tone: string = (body.tone || 'helpful').toString();
  const time: string = (body.time || 'week').toString();

  const invalid = validateQuery(query);
  if (invalid) return NextResponse.json({ error: invalid }, { status: 400 });

  // Client-selected post budget, clamped server-side to the allowed set so a
  // hand-crafted request cannot ask for thousands of model forward passes.
  const maxPosts = clampPostLimit(body.limit);

  // ── Retrieve ──────────────────────────────────────────────────────────
  // Fetch across two sort orders in parallel to broaden coverage. Reddit
  // often ranks "new" and "relevance" quite differently. Each leg fetches
  // roughly half the budget.
  const perSort = Math.max(5, Math.ceil(maxPosts / 2));
  const [rNew, rRel] = await Promise.all([
    searchReddit(query, { sort: 'new', time, limit: perSort }),
    searchReddit(query, { sort: 'relevance', time, limit: perSort }),
  ]);
  // Dedupe by id, and also by title+author: Reddit cross-posts appear in
  // several subreddits with different ids, which otherwise fills the results
  // with the same post repeated.
  const seenIds = new Set<string>();
  const seenPosts = new Set<string>();
  const dedupe = (list: typeof rNew) =>
    list.filter((p) => {
      const key = `${p.title.trim().toLowerCase()}|${p.author.toLowerCase()}`;
      if (seenIds.has(p.id) || seenPosts.has(key)) return false;
      seenIds.add(p.id);
      seenPosts.add(key);
      return true;
    });

  let raw = dedupe([...rNew, ...rRel]);

  // Nothing came back: widen the time window before giving up. A narrow query
  // over "past week" often has no matches while "past year" does.
  let widenedWindow = false;
  if (raw.length === 0 && time !== 'year') {
    widenedWindow = true;
    const [wNew, wRel] = await Promise.all([
      searchReddit(query, { sort: 'new', time: 'year', limit: perSort }),
      searchReddit(query, { sort: 'relevance', time: 'year', limit: perSort }),
    ]);
    raw = dedupe([...wNew, ...wRel]);
  }

  raw = raw.slice(0, maxPosts);

  if (raw.length === 0) {
    // Genuinely nothing retrieved. Reddit returned no matches, or rate
    // limited us. Say so plainly instead of implying the models found
    // nothing worth showing.
    return NextResponse.json({
      posts: [],
      stats: { fetched: 0, limit: maxPosts, tier: 'no_results', widenedWindow },
      reason:
        'Reddit returned no posts for this query. Try different wording, a longer time window, or wait a moment if requests are being rate limited.',
      quota: { remaining: rl.remaining, limit: rl.limit },
    });
  }

  // ── Classify intent + score relevance (in parallel per post) ─────────
  const scored = await Promise.all(
    raw.map(async (p) => {
      const [intent, rel] = await Promise.all([
        classifyIntent(p.title, p.selftext),
        scoreRelevance(query, p.title, p.selftext),
      ]);
      return {
        ...p,
        intent: intent?.label ?? 'off_topic',
        intent_confidence: intent?.confidence ?? 0,
        relevance: rel?.score ?? 0,
      };
    }),
  );

  // ── Filter, relaxing in stages rather than returning nothing ─────────
  // A strict filter is right when the corpus is rich, but on a narrow query
  // it can eliminate every candidate and leave the user staring at an error.
  // Widen the net step by step and report which tier produced the results, so
  // the UI can be honest about how they were selected.
  const byRelevance = (a: typeof scored[number], b: typeof scored[number]) =>
    b.relevance - a.relevance;

  const TIERS: Array<{ name: string; pick: () => typeof scored }> = [
    {
      // Ideal: clear buying intent and topically relevant.
      name: 'strict',
      pick: () =>
        scored.filter(
          (p) =>
            (p.intent === 'buying_intent' || p.intent === 'advice_seeking') &&
            p.relevance >= RELEVANCE_THRESHOLD,
        ),
    },
    {
      // Right intent, weaker topical match.
      name: 'intent_only',
      pick: () =>
        scored.filter(
          (p) => p.intent === 'buying_intent' || p.intent === 'advice_seeking',
        ),
    },
    {
      // Relevant to the query even if the intent classifier disagreed.
      name: 'relevance_only',
      pick: () => scored.filter((p) => p.relevance >= RELEVANCE_THRESHOLD / 2),
    },
    {
      // Last resort: the closest matches we retrieved, ranked.
      name: 'best_effort',
      pick: () => scored.slice(),
    },
  ];

  let survivors: typeof scored = [];
  let tier = 'strict';
  for (const t of TIERS) {
    const picked = t.pick().sort(byRelevance);
    if (picked.length > 0) {
      survivors = picked.slice(0, maxPosts);
      tier = t.name;
      break;
    }
  }

  // ── Sentiment + role + reply for surviving leads (parallel) ──────────
  const enriched = await Promise.all(
    survivors.map(async (p) => {
      const text = `${p.title}\n${p.selftext}`.slice(0, 2000);
      const [sent, role, rep] = await Promise.all([
        predictSentiment(text),
        classifyRole(text),
        generateReply(query, p.title, p.selftext, tone),
      ]);
      return {
        ...p,
        sentiment: sent?.sentiment ?? 'neutral',
        urgency: sent?.urgency ?? 'low',
        role: role?.role ?? 'other',
        reply: rep?.reply ?? '',
      };
    }),
  );

  return NextResponse.json({
    posts: enriched,
    stats: {
      fetched: raw.length,
      limit: maxPosts,
      passed_intent: scored.filter((p) => p.intent === 'buying_intent').length,
      passed_relevance: survivors.length,
      final: enriched.length,
      // Which filter tier produced these results. "strict" means they met
      // both the intent and relevance bars; anything else means the filter
      // was widened to avoid returning an empty list.
      tier,
    },
    quota: { remaining: rl.remaining, limit: rl.limit },
  });
}
