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
  const seen = new Set<string>();
  const raw = [...rNew, ...rRel].filter((p) => {
    if (seen.has(p.id)) return false;
    seen.add(p.id);
    return true;
  }).slice(0, maxPosts);

  if (raw.length === 0) {
    return NextResponse.json({
      posts: [],
      stats: { fetched: 0, limit: maxPosts },
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

  // ── Filter: keep buying-intent OR strongly relevant posts ────────────
  const survivors = scored
    .filter(
      (p) =>
        (p.intent === 'buying_intent' || p.intent === 'advice_seeking') &&
        p.relevance >= RELEVANCE_THRESHOLD,
    )
    .sort((a, b) => b.relevance - a.relevance);

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
    },
    quota: { remaining: rl.remaining, limit: rl.limit },
  });
}
