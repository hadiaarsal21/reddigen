// Direct Reddit fetch for the local build. No proxy, no credentials.
//
// Retrieval strategy: RSS first, JSON second.
//
// Reddit now answers 403 to anonymous requests on every *.json endpoint
// (search.json, r/<sub>/new.json, about.json, and oauth.reddit.com alike).
// The Atom/RSS feeds still return 200 for anonymous clients, so they are
// the primary path here. The JSON endpoints are kept as a fallback: they
// carry richer data (score, comment counts, clean selftext) and will be
// used automatically wherever they are reachable — a different network,
// a relaxed policy, or a future authenticated setup.
//
// Both paths funnel through fetchWithRetry, which backs off on 429. RSS
// is rate-limited too, so keep request volume modest.

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

// ── Atom/RSS helpers ────────────────────────────────────────────────────
// Reddit's feeds are Atom. Post entries look like:
//   <entry>
//     <author><name>/u/someone</name></author>
//     <content type="html">&lt;div&gt;selftext…&lt;/div&gt;</content>
//     <id>t3_abc123</id>
//     <link href="https://www.reddit.com/r/sub/comments/abc123/slug/" />
//     <updated>2026-01-01T00:00:00+00:00</updated>
//     <title>Post title</title>
//   </entry>
// Search feeds also return t5_ (subreddit) entries, which we drop.

function decodeOnce(s: string): string {
  return s
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(parseInt(d, 10)))
    .replace(/&amp;/g, '&'); // last: turns &amp;#39; into &#39; for the next pass
}

/**
 * Reddit's Atom <content> is XML-escaped HTML which itself contains HTML
 * entities, so the payload is effectively double-encoded (&amp;#39; →
 * &#39; → '). Decode repeatedly until the string stops changing.
 */
function decodeEntities(s: string): string {
  let prev = s;
  for (let i = 0; i < 3; i++) {
    const next = decodeOnce(prev);
    if (next === prev) break;
    prev = next;
  }
  return prev;
}

/** Strip tags from Reddit's HTML content and drop its trailing nav links. */
function htmlToText(html: string): string {
  return decodeEntities(html)
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n')
    .replace(/<[^>]+>/g, '')
    .replace(/\[link\]|\[comments\]/g, '')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function tag(xml: string, name: string): string {
  const m = xml.match(new RegExp(`<${name}[^>]*>([\\s\\S]*?)</${name}>`));
  return m ? m[1].trim() : '';
}

/** Parse an Atom feed into RedditPost[]. Fields absent from RSS default to 0. */
function parsePostsFromAtom(xml: string): RedditPost[] {
  const out: RedditPost[] = [];
  for (const m of xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)) {
    const e = m[1];

    // t3_ = link/post. Skip t5_ (subreddit) entries search feeds mix in.
    const rawId = tag(e, 'id');
    if (!rawId.startsWith('t3_')) continue;
    const id = rawId.slice(3);

    const href = e.match(/<link[^>]*href="([^"]+)"/)?.[1] ?? '';
    const permalink = decodeEntities(href);

    // Derive the subreddit from the permalink: works on search feeds and
    // subreddit feeds alike, unlike the per-entry <category> tag.
    const subreddit = permalink.match(/reddit\.com\/r\/([^/]+)/)?.[1] ?? 'unknown';

    const author = tag(e, 'name').replace(/^\/u\//, '') || 'unknown';
    const updated = tag(e, 'updated') || tag(e, 'published');
    const createdMs = updated ? Date.parse(updated) : NaN;

    out.push({
      id,
      title: decodeEntities(tag(e, 'title')),
      selftext: htmlToText(tag(e, 'content')).substring(0, 2000),
      author,
      subreddit,
      url: permalink,
      permalink,
      created_utc: Number.isNaN(createdMs) ? 0 : Math.floor(createdMs / 1000),
      // Not exposed by RSS. The JSON path fills these in when reachable.
      score: 0,
      num_comments: 0,
    });
  }
  return out;
}

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
  const qs =
    `q=${encodeURIComponent(query)}&sort=${sort}&t=${time}&limit=${limit}`;

  // ── Primary: RSS ──────────────────────────────────────────────────────
  const rss = await fetchWithRetry(`https://www.reddit.com/search.rss?${qs}`);
  if (rss?.ok) {
    try {
      const posts = parsePostsFromAtom(await rss.text());
      if (posts.length) return posts;
      console.warn(`[reddit] rss returned no post entries for "${query}"`);
    } catch (err) {
      console.warn('[reddit] rss parse failed', err);
    }
  } else {
    console.warn(`[reddit] rss search failed for "${query}": ${rss?.status ?? 'network'}`);
  }

  // ── Fallback: JSON (richer data where reachable) ──────────────────────
  const res = await fetchWithRetry(`https://www.reddit.com/search.json?${qs}`);
  if (!res || !res.ok) {
    console.warn(`[reddit] json search failed for "${query}": ${res?.status ?? 'network'}`);
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
  // ── Primary: RSS comment feed ─────────────────────────────────────────
  const rss = await fetchWithRetry(
    `https://www.reddit.com/r/${subreddit}/comments/${postId}.rss?limit=${limit}&sort=new`,
  );
  if (rss?.ok) {
    try {
      const xml = await rss.text();
      const feedTitle = decodeEntities(
        xml.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.trim() ?? '',
      );
      const out: RedditComment[] = [];
      for (const m of xml.matchAll(/<entry>([\s\S]*?)<\/entry>/g)) {
        const e = m[1];
        const rawId = tag(e, 'id');
        // t1_ = comment. The feed's first entry is the post itself (t3_).
        if (!rawId.startsWith('t1_')) continue;
        const body = htmlToText(tag(e, 'content'));
        if (!body || body === '[deleted]' || body === '[removed]') continue;
        if (body.length < 10) continue;
        const href = decodeEntities(e.match(/<link[^>]*href="([^"]+)"/)?.[1] ?? '');
        out.push({
          id: rawId.slice(3),
          author: tag(e, 'name').replace(/^\/u\//, '') || 'unknown',
          body: body.substring(0, 1500),
          score: 0, // not exposed by RSS
          post_id: postId,
          post_title: feedTitle,
          post_permalink: href,
          subreddit,
        });
      }
      if (out.length) return out;
    } catch {
      /* fall through to JSON */
    }
  }

  // ── Fallback: JSON — returns [post_listing, comments_listing] ─────────
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

  // ── Primary: RSS. The feed header carries the display title and the
  // subreddit description, but not subscriber/active counts. ────────────
  const rss = await fetchWithRetry(`https://www.reddit.com/r/${clean}/.rss?limit=1`);
  if (rss?.ok) {
    try {
      const xml = await rss.text();
      // Feed-level <title>/<subtitle> appear before the first <entry>.
      const head = xml.split('<entry>')[0];
      const title = decodeEntities(
        head.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.trim() ?? '',
      );
      const subtitle = htmlToText(
        head.match(/<subtitle[^>]*>([\s\S]*?)<\/subtitle>/)?.[1] ?? '',
      );
      if (title) {
        return {
          name: clean,
          title: title.replace(/^r\/[^:]*:\s*/, ''),
          description: subtitle.substring(0, 500),
          subscribers: 0, // not exposed by RSS
          active_users: 0,
        };
      }
    } catch {
      /* fall through to JSON */
    }
  }

  // ── Fallback: JSON (has subscriber + active counts) ───────────────────
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
