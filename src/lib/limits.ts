// Request limits shared by the UI and the API routes.
//
// Two independent protections:
//   1. Post-count caps  — bound how much work a single request can trigger.
//   2. Rate limiting    — bound how often requests can be made.
//
// The rate limiter is an in-process sliding window. That is the right scope
// for this build (one local Next.js server), but it resets on restart and is
// not shared across instances — a multi-instance deploy would need Redis.

/** Selectable post counts. The UI renders these; the API clamps to them. */
export const POST_LIMIT_OPTIONS = [5, 10, 25, 50, 75, 100] as const;

export const DEFAULT_POST_LIMIT = 50;

/** Hard ceiling. Even a hand-crafted request cannot exceed this. */
export const MAX_POST_LIMIT = 100;

export const MIN_QUERY_LENGTH = 3;
export const MAX_QUERY_LENGTH = 200;

/**
 * Coerce any client-supplied limit to the nearest allowed option.
 * Non-numeric, negative and oversized values collapse to safe defaults.
 */
export function clampPostLimit(raw: unknown): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return DEFAULT_POST_LIMIT;
  const floored = Math.floor(n);
  if (floored <= 0) return POST_LIMIT_OPTIONS[0];
  if (floored >= MAX_POST_LIMIT) return MAX_POST_LIMIT;
  // snap up to the next allowed option so the cap is never exceeded
  return POST_LIMIT_OPTIONS.find((o) => o >= floored) ?? DEFAULT_POST_LIMIT;
}

/** Validate a free-text query. Returns an error string, or null when valid. */
export function validateQuery(raw: unknown): string | null {
  if (typeof raw !== 'string') return 'Query must be a string';
  const q = raw.trim();
  if (q.length < MIN_QUERY_LENGTH) {
    return `Query too short — minimum ${MIN_QUERY_LENGTH} characters`;
  }
  if (q.length > MAX_QUERY_LENGTH) {
    return `Query too long — maximum ${MAX_QUERY_LENGTH} characters`;
  }
  return null;
}

// ── Sliding-window rate limiter ─────────────────────────────────────────

interface Window {
  hits: number[];
}

const buckets = new Map<string, Window>();

export interface RateRule {
  /** Max requests allowed inside the window. */
  max: number;
  /** Window length in milliseconds. */
  windowMs: number;
}

/**
 * Per-route budgets. Deep Scan and Discover are far more expensive than a
 * plain search — each fans out into many upstream Reddit calls — so they get
 * tighter budgets. These also keep us well inside Reddit's own rate limits,
 * which return 429 when hit too fast.
 */
export const RATE_RULES: Record<string, RateRule> = {
  search: { max: 10, windowMs: 60_000 },     // 10/min
  'deep-scan': { max: 3, windowMs: 60_000 }, // 3/min — heaviest
  discover: { max: 5, windowMs: 60_000 },    // 5/min
};

export interface RateResult {
  ok: boolean;
  remaining: number;
  /** Seconds until the window frees up. 0 when ok. */
  retryAfter: number;
  limit: number;
}

/** Record a hit for `key` and report whether it is allowed. */
export function rateLimit(key: string, rule: RateRule): RateResult {
  const now = Date.now();
  const cutoff = now - rule.windowMs;

  let bucket = buckets.get(key);
  if (!bucket) {
    bucket = { hits: [] };
    buckets.set(key, bucket);
  }
  // drop expired hits
  bucket.hits = bucket.hits.filter((t) => t > cutoff);

  if (bucket.hits.length >= rule.max) {
    const oldest = bucket.hits[0];
    const retryAfter = Math.max(1, Math.ceil((oldest + rule.windowMs - now) / 1000));
    return { ok: false, remaining: 0, retryAfter, limit: rule.max };
  }

  bucket.hits.push(now);

  // opportunistic cleanup so the map cannot grow without bound
  if (buckets.size > 500) {
    for (const [k, v] of buckets) {
      if (v.hits.every((t) => t <= cutoff)) buckets.delete(k);
    }
  }

  return {
    ok: true,
    remaining: rule.max - bucket.hits.length,
    retryAfter: 0,
    limit: rule.max,
  };
}

/** Best-effort client identity for rate limiting. */
export function clientKey(req: Request, route: string): string {
  const h = req.headers;
  const ip =
    h.get('x-forwarded-for')?.split(',')[0]?.trim() ||
    h.get('x-real-ip') ||
    'local';
  return `${route}:${ip}`;
}
