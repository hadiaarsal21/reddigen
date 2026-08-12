// POST /api/discover
//
// Given a product description, discovers the most relevant subreddits by:
//   1. Searching Reddit broadly for the topic
//   2. Grouping results by subreddit + counting activity
//   3. Fetching the subreddit's public description (metadata)
//   4. Semantic-ranking each subreddit's description against the query
//      using the relevance model (Sentence-BERT bi-encoder)
//   5. Combining semantic score + activity count for a final ranking
//
// This is the ONLY feature that uses the relevance ranker in a
// document-retrieval mode rather than pointwise scoring.

import { NextRequest, NextResponse } from 'next/server';
import { searchReddit, fetchSubredditInfo } from '@/lib/reddit';
import { scoreRelevance } from '@/lib/mlClient';

export const dynamic = 'force-dynamic';

const MAX_SUBREDDITS_TO_RANK = 25;
const MIN_POSTS_PER_SUBREDDIT = 1;

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const product: string = (body.product || body.query || '').toString().trim();
  if (product.length < 3) {
    return NextResponse.json({ error: 'Query too short' }, { status: 400 });
  }

  // ── Step 1: broad Reddit search (two sorts to widen coverage) ────────
  const [byNew, byRel] = await Promise.all([
    searchReddit(product, { sort: 'new', time: 'month', limit: 50 }),
    searchReddit(product, { sort: 'relevance', time: 'month', limit: 50 }),
  ]);

  const seen = new Set<string>();
  const posts = [...byNew, ...byRel].filter((p) => {
    if (seen.has(p.id)) return false;
    seen.add(p.id);
    return true;
  });

  if (posts.length === 0) {
    return NextResponse.json({
      subreddits: [],
      stats: { posts_examined: 0, subreddits_found: 0 },
    });
  }

  // ── Step 2: group by subreddit + count mentions ──────────────────────
  const bucket = new Map<string, { name: string; mentions: number; sample_titles: string[] }>();
  for (const p of posts) {
    if (!p.subreddit || p.subreddit === 'unknown') continue;
    const cur = bucket.get(p.subreddit) ?? {
      name: p.subreddit,
      mentions: 0,
      sample_titles: [],
    };
    cur.mentions++;
    if (cur.sample_titles.length < 3) cur.sample_titles.push(p.title);
    bucket.set(p.subreddit, cur);
  }

  const grouped = Array.from(bucket.values())
    .filter((s) => s.mentions >= MIN_POSTS_PER_SUBREDDIT)
    .sort((a, b) => b.mentions - a.mentions)
    .slice(0, MAX_SUBREDDITS_TO_RANK);

  // ── Step 3: fetch metadata for each candidate subreddit (parallel) ───
  const metas = await Promise.all(grouped.map((s) => fetchSubredditInfo(s.name)));

  // ── Step 4: semantic-rank each subreddit's description ───────────────
  const scored = await Promise.all(
    grouped.map(async (s, i) => {
      const meta = metas[i];
      const desc = meta?.description ?? '';
      const semanticScore = desc
        ? (await scoreRelevance(product, meta?.title ?? s.name, desc))?.score ?? 0
        : 0;
      // Combined score: 70% semantic, 30% activity (log-scaled)
      const activityBoost = Math.log(1 + s.mentions) / Math.log(1 + 50);
      const composite = 0.7 * semanticScore + 0.3 * activityBoost;
      return {
        name: s.name,
        title: meta?.title || s.name,
        description: desc,
        subscribers: meta?.subscribers ?? 0,
        active_users: meta?.active_users ?? 0,
        mentions: s.mentions,
        sample_titles: s.sample_titles,
        semantic_score: semanticScore,
        composite_score: composite,
        url: `https://www.reddit.com/r/${s.name}`,
      };
    }),
  );

  scored.sort((a, b) => b.composite_score - a.composite_score);

  return NextResponse.json({
    subreddits: scored,
    stats: {
      posts_examined: posts.length,
      subreddits_found: scored.length,
    },
  });
}
