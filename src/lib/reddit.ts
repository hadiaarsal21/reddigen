// Direct Reddit fetch for the local build. No proxy — for a localhost
// demo running from a home / dev machine, Reddit's public RSS + JSON
// endpoints work fine when hit at a modest rate.

const USER_AGENTS = [
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
];

function randomUA(): string {
  return USER_AGENTS[Math.floor(Math.random() * USER_AGENTS.length)];
}

export interface RedditPost {
  id: string;
  title: string;
  selftext: string;
  author: string;
  subreddit: string;
  url: string;
  permalink: string;
  created_utc: number;
  score: number;
  num_comments: number;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function fetchWithRetry(url: string, retries = 2): Promise<Response | null> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url, {
        headers: {
          'User-Agent': randomUA(),
          Accept: 'application/json, application/rss+xml, */*',
          'Accept-Language': 'en-US,en;q=0.9',
        },
        signal: AbortSignal.timeout(15000),
      });
      if (res.status === 429) {
        await sleep(3000 * Math.pow(2, attempt));
        continue;
      }
      return res;
    } catch {
      if (attempt < retries) await sleep(2000 * (attempt + 1));
    }
  }
  return null;
}

/**
 * Fetch Reddit search results for a query. Returns raw posts before any
 * ML scoring — the search route handler is responsible for calling the
 * ML server to filter and rank them.
 */
export async function searchReddit(
  query: string,
  options: { sort?: string; time?: string; limit?: number } = {},
): Promise<RedditPost[]> {
  const { sort = 'new', time = 'week', limit = 50 } = options;
  const url =
    `https://www.reddit.com/search.json?q=${encodeURIComponent(query)}` +
    `&sort=${sort}&t=${time}&limit=${limit}`;

  const res = await fetchWithRetry(url);
  if (!res || !res.ok) {
    console.warn(`[reddit] search failed for "${query}": ${res?.status ?? 'network'}`);
    return [];
  }

  try {
    const data = await res.json();
    const children: unknown[] = data?.data?.children ?? [];
    return children
      .filter((c: any) => c?.kind === 't3')
      .map((c: any) => {
        const d = c.data;
        return {
          id: d.id,
          title: d.title || '',
          selftext: (d.selftext || '').substring(0, 2000),
          author: d.author || 'unknown',
          subreddit: d.subreddit || 'unknown',
          url: d.url || '',
          permalink: `https://www.reddit.com${d.permalink || ''}`,
          created_utc: d.created_utc || 0,
          score: d.score || 0,
          num_comments: d.num_comments || 0,
        } as RedditPost;
      });
  } catch (err) {
    console.warn('[reddit] JSON parse failed', err);
    return [];
  }
}
