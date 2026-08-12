// HTTP client for the local FastAPI ML server. Every trained model in
// this project is served through this single indirection so the
// Next.js app stays entirely framework-agnostic on the ML side.

const BASE = process.env.ML_SERVICE_URL || 'http://localhost:8000';

async function post<TReq, TRes>(path: string, body: TReq): Promise<TRes | null> {
  try {
    const res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) {
      console.warn(`[ml] ${path} → HTTP ${res.status}`);
      return null;
    }
    return (await res.json()) as TRes;
  } catch (err) {
    console.warn(`[ml] ${path} failed:`, err);
    return null;
  }
}

// ─── Endpoint types ──────────────────────────────────────────────────────

export interface IntentResp {
  label: 'buying_intent' | 'advice_seeking' | 'discussion' | 'off_topic';
  confidence: number;
}

export interface RelevanceResp {
  score: number;
}

export interface RoleResp {
  role: 'buyer' | 'seller' | 'advisor' | 'other';
  confidence: number;
}

export interface SentimentResp {
  sentiment: 'positive' | 'neutral' | 'negative';
  urgency: 'low' | 'medium' | 'high';
}

export interface ReplyResp {
  reply: string;
}

// ─── Public helpers ──────────────────────────────────────────────────────

export const classifyIntent = (title: string, body = '') =>
  post<{ title: string; body: string }, IntentResp>('/classify-intent', { title, body });

export const scoreRelevance = (query: string, title: string, body = '') =>
  post<{ query: string; title: string; body: string }, RelevanceResp>(
    '/score-relevance',
    { query, title, body },
  );

export const classifyRole = (text: string) =>
  post<{ text: string }, RoleResp>('/classify-role', { text });

export const predictSentiment = (text: string) =>
  post<{ text: string }, SentimentResp>('/predict-sentiment', { text });

export const generateReply = (
  query: string,
  postTitle: string,
  postBody: string,
  tone = 'helpful',
) =>
  post<
    { query: string; post_title: string; post_body: string; tone: string },
    ReplyResp
  >('/generate-reply', {
    query,
    post_title: postTitle,
    post_body: postBody,
    tone,
  });

export async function isServerReachable(): Promise<boolean> {
  try {
    const res = await fetch(BASE, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}
