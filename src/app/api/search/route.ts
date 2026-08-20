// POST /api/search
//
// The single endpoint the dashboard hits. Retrieves Reddit posts for the
// query, then chains all five trained models to produce a scored, ranked,
// reply-drafted lead list.
//
// Two guarantees shape the design:
//
//   1. The user asks for a NUMBER OF LEADS, not a number of posts to scan.
//      We keep retrieving from different angles until that many qualifying
//      leads exist, rather than scanning a fixed batch and returning
//      whatever happens to survive.
//
//   2. Every returned lead must be genuinely on-topic AND show buying
//      intent. Nothing is padded in to reach the number. If Reddit does not
//      hold enough matching posts, we return fewer and say so, because a
//      list padded with junk is worse than a short list.

import { NextRequest, NextResponse } from 'next/server';
import { searchReddit, type RedditPost } from '@/lib/reddit';
import {
  clampPostLimit,
  clientKey,
  rateLimit,
  RATE_RULES,
  validateQuery,
} from '@/lib/limits';
import {
  hasBuyingSignal,
  leadScore,
  looksLikeJobSeeker,
  looksLikeSeller,
  qualifies,
  rejectionReason,
  titleFingerprint,
  topicOverlap,
  type Rejection,
} from '@/lib/leadGate';
import {
  classifyIntent,
  scoreRelevance,
  predictSentiment,
  classifyRole,
  generateReply,
} from '@/lib/mlClient';

export const dynamic = 'force-dynamic';

/**
 * Ceiling on how many posts we will put through the intent + relevance
 * models while hunting for the target. Each costs two model calls, so this
 * bounds the work a single request can trigger regardless of how many
 * rounds it takes.
 */
const MAX_SCORED = 320;

/** Stop after this many consecutive rounds that add no new posts. */
const MAX_DRY_ROUNDS = 2;

/**
 * How many posts to score at once.
 *
 * The ML server runs five models on one CPU. Firing a whole round at it in
 * parallel (100 posts x 3 model calls) swamps it: requests queue past their
 * timeout, the client turns each timeout into null, and the route reads that
 * as "off_topic" — so a search silently returned nothing while the models
 * were perfectly healthy. Small batches keep every call inside its budget.
 */
const SCORE_CONCURRENCY = 6;

/** Run an async mapper over items, at most `limit` in flight at a time. */
async function mapLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      out[i] = await fn(items[i]);
    }
  });
  await Promise.all(workers);
  return out;
}

type Scored = RedditPost & {
  intent: string;
  intent_confidence: number;
  relevance: number;
  overlap: number;
  role: string;
  role_confidence: number;
  sellerFlair: boolean;
  jobSeeker: boolean;
  buyingSignal: boolean;
};

/**
 * Retrieval angles, tried in order until the target is met.
 *
 * Reddit's own ranking differs sharply between sorts and time windows, so
 * each combination surfaces posts the others miss. The later rounds add
 * buying-intent phrasings to the query itself, which is the most effective
 * way to find more posts that will actually pass the gate rather than more
 * posts in general.
 */
function retrievalPlan(query: string, time: string) {
  const widerTimes = ['month', 'year', 'all'].filter((t) => t !== time);
  return [
    { q: query, sort: 'relevance', time },
    { q: query, sort: 'new', time },
    ...widerTimes.map((t) => ({ q: query, sort: 'relevance', time: t })),
    { q: `looking for ${query}`, sort: 'relevance', time: 'year' },
    { q: `need ${query}`, sort: 'relevance', time: 'year' },
    { q: `hiring ${query}`, sort: 'relevance', time: 'year' },
    { q: `recommend ${query}`, sort: 'relevance', time: 'year' },
    { q: `${query} freelancer`, sort: 'relevance', time: 'year' },
    { q: query, sort: 'new', time: 'year' },
  ];
}

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

  // How many LEADS the user asked for. Clamped server-side so a hand-built
  // request cannot ask for thousands.
  const target = clampPostLimit(body.limit);

  // ── Retrieve and qualify, round by round ─────────────────────────────
  const seenIds = new Set<string>();
  const seenPosts = new Set<string>();
  const qualified: Scored[] = [];
  const rejections: Record<string, number> = {
    intent: 0, confidence: 0, relevance: 0, offtopic: 0, seller: 0, jobseeker: 0, nodemand: 0,
  };

  let fetched = 0;
  let scoredCount = 0;
  let dryRounds = 0;
  let roundsUsed = 0;
  let emptyFetches = 0;

  const plan = retrievalPlan(query, time);

  for (const round of plan) {
    if (qualified.length >= target) break;
    if (scoredCount >= MAX_SCORED) break;
    if (dryRounds >= MAX_DRY_ROUNDS) break;

    // Space the rounds out. Firing nine searches back to back is what
    // trips Reddit's rate limiter, which then returns nothing at all.
    if (roundsUsed > 0) await new Promise((r) => setTimeout(r, 400));
    roundsUsed++;

    const batch = await searchReddit(round.q, {
      sort: round.sort,
      time: round.time,
      limit: 100, // ask Reddit for as much as it will give in one call
    });
    fetched += batch.length;

    // Drop anything already seen. Three keys are needed: the id, the exact
    // title+author for plain cross-posts, and a word-set fingerprint for the
    // same advert reposted with the title reordered.
    const fresh = batch.filter((p) => {
      const exact = `${p.title.trim().toLowerCase()}|${p.author.toLowerCase()}`;
      const fingerprint = titleFingerprint(p.title);
      if (
        seenIds.has(p.id) ||
        seenPosts.has(exact) ||
        (fingerprint.length > 0 && seenPosts.has(fingerprint))
      ) {
        return false;
      }
      seenIds.add(p.id);
      seenPosts.add(exact);
      if (fingerprint.length > 0) seenPosts.add(fingerprint);
      return true;
    });

    // Two different kinds of empty round, which must not be treated alike:
    //
    //   batch empty      Reddit gave us nothing, usually a transient 429.
    //                    A different query or time window may still work, so
    //                    keep going rather than concluding there is no data.
    //   batch non-empty  We got posts but had seen them all. That is a real
    //                    sign this angle is exhausted.
    if (batch.length === 0) {
      emptyFetches++;
      continue;
    }
    if (fresh.length === 0) {
      dryRounds++;
      continue;
    }
    dryRounds = 0;

    // Only score as many as the budget still allows.
    const room = Math.max(0, MAX_SCORED - scoredCount);
    const toScore = fresh.slice(0, room);
    scoredCount += toScore.length;

    const scored: Scored[] = await mapLimit(toScore, SCORE_CONCURRENCY, async (p) => {
        // The flair check is free, and a post advertising services can be
        // rejected without spending three model calls on it.
        const sellerFlair = looksLikeSeller(p.title, p.selftext);
        const jobSeeker = looksLikeJobSeeker(p.title, p.selftext);
        const buyingSignal = hasBuyingSignal(p.title, p.selftext);
        const text = `${p.title}\n${p.selftext ?? ''}`.slice(0, 2000);

        if (sellerFlair) {
          return {
            ...p,
            intent: 'off_topic',
            intent_confidence: 0,
            relevance: 0,
            overlap: 0,
            role: 'seller',
            role_confidence: 1,
            sellerFlair: true,
            jobSeeker,
            buyingSignal,
          };
        }

        // Role runs here rather than during enrichment so competitors are
        // excluded before they can occupy a slot in the results.
        const [intent, rel, role] = await Promise.all([
          classifyIntent(p.title, p.selftext),
          scoreRelevance(query, p.title, p.selftext),
          classifyRole(text),
        ]);
        return {
          ...p,
          intent: intent?.label ?? 'off_topic',
          intent_confidence: intent?.confidence ?? 0,
          relevance: rel?.score ?? 0,
          overlap: topicOverlap(query, `${p.title} ${p.selftext ?? ''}`),
          role: role?.role ?? 'other',
          role_confidence: role?.confidence ?? 0,
          sellerFlair: false,
          jobSeeker,
          buyingSignal,
        };
    });

    for (const s of scored) {
      if (qualifies(s)) {
        qualified.push(s);
      } else {
        const why = rejectionReason(s) as Exclude<Rejection, null>;
        rejections[why] = (rejections[why] ?? 0) + 1;
      }
    }
  }

  // Best leads first, then cut to exactly what was asked for.
  qualified.sort((a, b) => leadScore(b) - leadScore(a));
  const selected = qualified.slice(0, target);

  if (selected.length === 0) {
    return NextResponse.json({
      posts: [],
      stats: {
        requested: target,
        delivered: 0,
        fetched,
        scored: scoredCount,
        rounds: roundsUsed,
        emptyFetches,
        rejections,
      },
      reason:
        fetched === 0
          ? 'Reddit returned no posts for this query. Try different wording, or wait a moment if requests are being rate limited.'
          : `Checked ${scoredCount} posts and none showed clear buying intent for this topic. Try wording it the way a customer would ask, or widen the time window.`,
      quota: { remaining: rl.remaining, limit: rl.limit },
    });
  }

  // ── Enrich only the leads we are actually returning ──────────────────
  // Sentiment, role and especially reply generation are the expensive
  // calls, so they run on the final set rather than every candidate.
  // Reply generation is the slowest call in the system, so this runs at a
  // low concurrency too rather than launching every generation at once.
  const enriched = await mapLimit(selected, 3, async (p) => {
    const text = `${p.title}\n${p.selftext}`.slice(0, 2000);
    // Role was already decided during qualification, so it is not recomputed.
    const [sent, rep] = await Promise.all([
      predictSentiment(text),
      generateReply(query, p.title, p.selftext, tone),
    ]);
    return {
      ...p,
      sentiment: sent?.sentiment ?? 'neutral',
      urgency: sent?.urgency ?? 'low',
      reply: rep?.reply ?? '',
    };
  });

  return NextResponse.json({
    posts: enriched,
    stats: {
      requested: target,
      delivered: enriched.length,
      exact: enriched.length === target,
      fetched,
      scored: scoredCount,
      rounds: roundsUsed,
      emptyFetches,
      qualified: qualified.length,
      rejections,
      // Kept so older clients reading these keys still work.
      final: enriched.length,
      limit: target,
    },
    quota: { remaining: rl.remaining, limit: rl.limit },
  });
}
