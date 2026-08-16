// POST /api/deep-scan
//
// Given a product description, mines COMMENT THREADS on Reddit for buyer
// signals hiding under other people's posts. This is where the real
// value hides: someone posts "[For Hire] SEO expert" and 40 people reply
// with "I need this too!" — those replies are qualified leads even
// though the original post wasn't about buying.
//
// Pipeline:
//   1. Find target posts likely to attract buyer replies (searches
//      "[for hire]", "hiring", "recommend" variants of the query)
//   2. Fetch the top comment thread for each promising post
//   3. Run every comment through the ROLE classifier (buyer/seller/advisor)
//   4. Keep only BUYER comments (with confidence threshold)
//   5. Optionally enrich with sentiment + a draft reply

import { NextRequest, NextResponse } from 'next/server';
import { searchReddit, fetchComments } from '@/lib/reddit';
import {
  clampPostLimit,
  clientKey,
  rateLimit,
  RATE_RULES,
  validateQuery,
} from '@/lib/limits';
import {
  classifyRole,
  predictSentiment,
  generateReply,
  scoreRelevance,
} from '@/lib/mlClient';

export const dynamic = 'force-dynamic';

const MAX_COMMENTS_PER_POST = 25;
const ROLE_CONFIDENCE_THRESHOLD = 0.55;

// Deep Scan fans out: every scanned post costs one comment-thread fetch plus
// a role classification per comment. Cap the posts hard so one request can't
// trigger hundreds of upstream calls.
const DEEP_SCAN_MAX_POSTS = 25;

// Buyer-attracting search variants — same product, different phrasings
// that tend to attract comments from people who want the service too.
function buildQueries(product: string): string[] {
  const q = product.toLowerCase().trim();
  return [
    `[for hire] ${q}`,
    `hiring ${q}`,
    `recommend ${q}`,
    `looking for ${q}`,
    `${q} freelancer`,
  ];
}

export async function POST(req: NextRequest) {
  const rl = rateLimit(clientKey(req, 'deep-scan'), RATE_RULES['deep-scan']);
  if (!rl.ok) {
    return NextResponse.json(
      {
        error: `Rate limit reached — ${rl.limit} deep scans per minute. Try again in ${rl.retryAfter}s.`,
        retryAfter: rl.retryAfter,
      },
      { status: 429, headers: { 'Retry-After': String(rl.retryAfter) } },
    );
  }

  const body = await req.json().catch(() => ({}));
  const product: string = (body.product || body.query || '').toString().trim();
  const tone: string = (body.tone || 'helpful').toString();

  const invalid = validateQuery(product);
  if (invalid) return NextResponse.json({ error: invalid }, { status: 400 });

  const maxPostsToScan = Math.min(clampPostLimit(body.limit), DEEP_SCAN_MAX_POSTS);

  // ── Step 1: search for posts that tend to attract buyer replies ──────
  const queries = buildQueries(product);
  const perQuery = Math.max(4, Math.ceil(maxPostsToScan / queries.length) * 2);
  const postArrays = await Promise.all(
    queries.map((q) => searchReddit(q, { sort: 'new', time: 'month', limit: perQuery })),
  );

  // Flatten + dedupe by post ID, then rank by num_comments (more
  // comments = more chance of finding real buyers)
  const seen = new Set<string>();
  const posts: any[] = [];
  for (const arr of postArrays) {
    for (const p of arr) {
      if (seen.has(p.id)) continue;
      seen.add(p.id);
      posts.push(p);
    }
  }
  posts.sort((a, b) => b.num_comments - a.num_comments);
  const targetPosts = posts.slice(0, maxPostsToScan);

  if (targetPosts.length === 0) {
    return NextResponse.json({
      leads: [],
      stats: { posts_scanned: 0, comments_examined: 0, buyers_found: 0 },
      quota: { remaining: rl.remaining, limit: rl.limit },
    });
  }

  // ── Step 2: fetch comments for each target post (in parallel) ────────
  const commentBundles = await Promise.all(
    targetPosts.map((p) =>
      fetchComments(p.id, p.subreddit, MAX_COMMENTS_PER_POST).then((cs) => ({
        post: p,
        comments: cs,
      })),
    ),
  );

  const allComments = commentBundles.flatMap((b) =>
    b.comments.map((c) => ({ ...c, parent_post: b.post })),
  );

  // ── Step 3: role classify every comment (in parallel batches) ────────
  const roleResults = await Promise.all(
    allComments.map(async (c) => {
      const role = await classifyRole(c.body);
      return { comment: c, role: role?.role ?? 'other', confidence: role?.confidence ?? 0 };
    }),
  );

  // ── Step 4: keep only high-confidence buyers ─────────────────────────
  const buyers = roleResults.filter(
    (r) => r.role === 'buyer' && r.confidence >= ROLE_CONFIDENCE_THRESHOLD,
  );

  // ── Step 5: enrich each buyer with sentiment, relevance to product,
  //           and a draft reply grounded in the parent post
  const enriched = await Promise.all(
    buyers.map(async (b) => {
      const c = b.comment;
      const [sent, rel, rep] = await Promise.all([
        predictSentiment(c.body),
        scoreRelevance(product, c.parent_post.title, c.body),
        generateReply(product, c.parent_post.title, c.body, tone),
      ]);
      return {
        // Comment identity
        id: `t1_${c.id}`,
        commentId: c.id,
        author: c.author,
        body: c.body,
        subreddit: c.subreddit,
        // Parent post context
        parent_post_id: c.parent_post.id,
        parent_post_title: c.parent_post.title,
        parent_post_url: c.parent_post.permalink,
        // ML outputs
        role: b.role,
        role_confidence: b.confidence,
        relevance: rel?.score ?? 0,
        sentiment: sent?.sentiment ?? 'neutral',
        urgency: sent?.urgency ?? 'low',
        reply: rep?.reply ?? '',
      };
    }),
  );

  // Rank buyer-leads by relevance-to-product
  enriched.sort((a, b) => b.relevance - a.relevance);

  return NextResponse.json({
    leads: enriched,
    stats: {
      posts_scanned: targetPosts.length,
      comments_examined: allComments.length,
      buyers_found: enriched.length,
      sellers_filtered: roleResults.filter((r) => r.role === 'seller').length,
      advisors_filtered: roleResults.filter((r) => r.role === 'advisor').length,
    },
    quota: { remaining: rl.remaining, limit: rl.limit },
  });
}
