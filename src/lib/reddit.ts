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

export interface RedditComment {
  id: string;
  author: string;
  body: string;
  score: number;
  post_id: string;
  post_title: string;
  post_permalink: string;
  subreddit: string;
}

export interface SubredditInfo {
  name: string;
  title: string;
  description: string;
  subscribers: number;
  active_users: number;
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

/**
 * Fetch top-level comments for a given post ID. Used by Deep Scan to
 * mine buyer signals from replies under sellers'/service providers' posts.
 */
export async function fetchComments(
  postId: string,
  subreddit: string,
  limit = 100,
): Promise<RedditComment[]> {
  // Reddit's JSON API returns [post_listing, comments_listing]
  const url =
    `https://www.reddit.com/r/${subreddit}/comments/${postId}.json` +
    `?limit=${limit}&sort=new`;
  const res = await fetchWithRetry(url);
  if (!res || !res.ok) return [];

  try {
    const data = await res.json();
    if (!Array.isArray(data) || data.length < 2) return [];
    const postData = data[0]?.data?.children?.[0]?.data;
    const commentTree: any[] = data[1]?.data?.children ?? [];
    const postTitle = postData?.title || '';
    const postPermalink = `https://www.reddit.com${postData?.permalink || ''}`;

    const out: RedditComment[] = [];
    function walk(nodes: any[]) {
      for (const n of nodes) {
        if (n?.kind !== 't1') continue;
        const d = n.data;
        const body = (d.body || '').trim();
        if (!body || body === '[deleted]' || body === '[removed]') continue;
        if (body.length < 10) continue;
        out.push({
          id: d.id,
          author: d.author || 'unknown',
          body: body.substring(0, 1500),
          score: d.score || 0,
          post_id: postId,
          post_title: postTitle,
          post_permalink: postPermalink,
          subreddit,
        });
        const replies = d.replies?.data?.children;
        if (Array.isArray(replies)) walk(replies);
      }
    }
    walk(commentTree);
    return out;
  } catch {
    return [];
  }
}

/**
 * Fetch metadata (title, description, subscriber count) for a subreddit.
 * Used by Discover to enrich the ranked list.
 */
export async function fetchSubredditInfo(name: string): Promise<SubredditInfo | null> {
  const clean = name.replace(/^r\//, '').trim();
  const res = await fetchWithRetry(`https://www.reddit.com/r/${clean}/about.json`);
  if (!res || !res.ok) return null;
  try {
    const data = await res.json();
    const d = data?.data;
    if (!d) return null;
    return {
      name: clean,
      title: d.title || '',
      description: (d.public_description || d.description || '').substring(0, 500),
      subscribers: d.subscribers || 0,
      active_users: d.active_user_count || 0,
    };
  } catch {
    return null;
  }
}
