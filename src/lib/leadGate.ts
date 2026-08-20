// Decides whether a scored post is genuinely a lead for this query.
//
// The intent model is trained on synthetic data, so it is confident but not
// always right on real Reddit text. A model-only gate let obvious nonsense
// through: a Halo game trailer was once offered as an "SEO agency" lead
// because the classifier happened to label it buying_intent.
//
// So qualification needs two independent signals that must BOTH hold:
//
//   1. Intent   — the post is someone trying to buy/hire, not discussing
//   2. Topic    — the post is actually about the thing being searched for
//
// The topic check is deliberately deterministic (word overlap) rather than
// another model. When a model is wrong, a second model is often wrong in the
// same direction; plain word matching fails independently, which is exactly
// what a backstop needs to do.

/** Words that carry no topical meaning and must not count as a match. */
const STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'for', 'to', 'of', 'in', 'on', 'at',
  'by', 'from', 'with', 'about', 'as', 'is', 'are', 'was', 'were', 'be',
  'been', 'being', 'this', 'that', 'these', 'those', 'it', 'its', 'i', 'we',
  'you', 'they', 'he', 'she', 'my', 'our', 'your', 'their', 'me', 'us',
  'looking', 'look', 'need', 'needs', 'needed', 'want', 'wants', 'wanted',
  'hiring', 'hire', 'find', 'finding', 'seeking', 'searching', 'search',
  'recommend', 'recommendation', 'recommendations', 'suggest', 'suggestions',
  'best', 'good', 'great', 'top', 'any', 'anyone', 'someone', 'somebody',
  'help', 'please', 'thanks', 'thank', 'hello', 'hi', 'hey',
  'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did', 'have',
  'has', 'had', 'get', 'got', 'make', 'made', 'know', 'like', 'want',
  'new', 'old', 'much', 'many', 'more', 'most', 'some', 'all', 'how', 'what',
  'when', 'where', 'who', 'why', 'which', 'there', 'here', 'not', 'no',
]);

/** Split text into meaningful lowercase words. */
export function tokenize(text: string): string[] {
  return (text || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s+#.-]/g, ' ')
    .split(/\s+/)
    .map((w) => w.replace(/^[.\-]+|[.\-]+$/g, ''))
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w));
}

/** Crude singular form, so "developers" matches "developer". */
function stem(word: string): string {
  if (word.length > 4 && word.endsWith('ies')) return `${word.slice(0, -3)}y`;
  if (word.length > 3 && word.endsWith('es')) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith('s') && !word.endsWith('ss')) {
    return word.slice(0, -1);
  }
  return word;
}

/**
 * What fraction of the meaningful words in the query appear in the post.
 * Returns 0 when the query has no meaningful words left after filtering
 * (e.g. "looking for someone"), which makes the caller fall back to the
 * semantic score alone rather than rejecting everything.
 */
export function topicOverlap(query: string, text: string): number {
  const qTokens = [...new Set(tokenize(query).map(stem))];
  if (qTokens.length === 0) return -1; // "no opinion"

  const postTokens = new Set(tokenize(text).map(stem));
  const hits = qTokens.filter((t) => postTokens.has(t)).length;
  return hits / qTokens.length;
}

/**
 * Reddit's own flair conventions, which are far more reliable than any model
 * for telling buyers from sellers.
 *
 * This matters because the two read almost identically to a classifier:
 * "[For Hire] Python Developer" and "[Hiring] Python Developer" share nearly
 * every word, but one is a competitor advertising and the other is a
 * customer. The intent model scores both as buying_intent with 0.99
 * confidence, so a competitor's advert lands at the top of the lead list.
 */
const SELLER_MARKERS = [
  /\[\s*for\s*hire\s*\]/i,
  /\[\s*offer(ing)?\s*\]/i,
  /\[\s*portfolio\s*\]/i,
  /\[\s*showcase\s*\]/i,
  /\[\s*available\s*\]/i,
  /\[\s*service[s]?\s*\]/i,
  /\[\s*promo(tion)?\s*\]/i,
  /\bhire\s+me\b/i,
  /\bmy\s+services\b/i,
  /\bi(?:'m| am)\s+(?:a\s+)?freelance/i,
  /\bopen\s+(?:for|to)\s+(?:work|commissions|clients)\b/i,
  /\bdm\s+me\s+(?:for|if)\b/i,
  /\bstarting\s+at\s+\$/i,
  /\bmy\s+rate[s]?\s+(?:is|are|start)/i,
];

/** Markers that positively identify a buyer, which override a seller guess. */
const BUYER_MARKERS = [
  /\[\s*hiring\s*\]/i,
  /\[\s*task\s*\]/i,
  /\[\s*request\s*\]/i,
  /\[\s*paid\s*\]/i,
  /\bwe\s+are\s+(?:hiring|looking)\b/i,
  /\blooking\s+to\s+(?:hire|pay|buy)\b/i,
  /\bmy\s+budget\s+is\b/i,
  /\bwilling\s+to\s+pay\b/i,
];

/**
 * Someone looking for employment, not looking to buy a service.
 *
 * This is the single biggest source of false leads. "Python developer" pulls
 * in people who want to BE one as much as people who want to HIRE one, and
 * the two are near-identical to a classifier: same nouns, same enthusiasm,
 * both plainly "about" the topic. The intent model scores them buying_intent
 * with high confidence because its training data contains no job hunters at
 * all.
 *
 * For someone selling a service these are the opposite of a customer, so they
 * are rejected outright rather than ranked lower.
 */
const JOB_SEEKER_MARKERS = [
  /\b(?:my|a|the)\s+first\s+(?:job|role|position|internship)\b/i,
  /\b(?:find|get|land)\s+(?:me\s+)?(?:a|my|the)\b[^.?!]{0,40}\b(?:job|role|position)\b/i,
  /\bjob\s+(?:hunt|hunting|search|seeking|seeker|market|application)\b/i,
  /\blooking\s+for\s+(?:a\s+)?(?:job|work|role|position|internship|opportunit)/i,
  /\b(?:resume|cv)\s+(?:review|feedback|tips|advice|help)\b/i,
  /\bmy\s+(?:resume|cv|portfolio)\b/i,
  /\binterview\s+(?:experience|question|prep|preparation|process|round|tips)/i,
  /\b(?:fresher|graduate|bootcamp|self[- ]taught)\b[^.?!]{0,50}\b(?:job|role|hired|placement|career)\b/i,
  /\bhow\s+(?:did|do)\s+you\s+(?:get|land|find|become)\b/i,
  /\bapplying\s+(?:for|to)\s+(?:a\s+)?(?:job|role|position)/i,
  /\bcareer\s+(?:advice|change|switch|path|transition)\b/i,
  /\b(?:tips|advice)\s+(?:for|on)\s+(?:getting|landing|becoming|breaking\s+into)\b/i,
  /\bcan'?t\s+(?:find|get)\s+(?:a\s+)?(?:job|work)\b/i,
  /\bunemployed\b/i,
  /\bwant\s+to\s+(?:become|be)\s+(?:a|an)\b/i,
  /\bshould\s+i\s+learn\b/i,
  /\bworth\s+learning\b/i,
  /\broadmap\b/i,
  /\bsalary\s+(?:expectation|range|for)\b/i,
];

/** True when the poster is job hunting or asking careers questions. */
export function looksLikeJobSeeker(title: string, body = ''): boolean {
  const text = `${title}\n${body}`;
  // An explicit hiring marker wins: "[Hiring] ... interview process" is a
  // company describing its own process, not a candidate asking about one.
  if (BUYER_MARKERS.some((re) => re.test(text))) return false;
  return JOB_SEEKER_MARKERS.some((re) => re.test(text));
}

/**
 * Phrases that show the poster actually wants to obtain something.
 *
 * Rejecting bad posts is not enough on its own. Plenty of posts are on-topic,
 * are not adverts, are not job hunting, and are still not leads: "What Python
 * habit did you stop doing?" is a discussion thread that ticks every negative
 * check. Requiring a positive signal is the difference between "nothing
 * disqualifies this" and "this person wants to buy".
 *
 * Deliberately broad, because a customer can phrase the same need many ways,
 * but every entry expresses wanting, needing, paying for or commissioning
 * something.
 */
const BUYING_SIGNALS = [
  // explicit hiring / commissioning
  /\[\s*(?:hiring|task|request|paid|for\s*sale)\s*\]/i,
  /\b(?:i|we|my\s+\w+)\s+(?:need|needs|require|am\s+looking\s+for|are\s+looking\s+for)\b/i,
  /\bneed\s+(?:a|an|some(?:one|body)|help|advice)\b/i,
  /\blooking\s+(?:for|to\s+hire|to\s+buy|to\s+pay)\b/i,
  /\b(?:want|wanted|wanting)\s+to\s+(?:hire|buy|pay|commission|outsource)\b/i,
  /\b(?:can|could)\s+(?:any(?:one|body)|some(?:one|body))\s+(?:help|build|make|do|recommend)\b/i,
  /\bany(?:one|body)\s+(?:know|recommend|offer|available|interested)\b/i,
  /\brecommend(?:ation|ations)?\s+(?:for|on)\b/i,
  /\bwho\s+can\s+(?:help|build|do|make)\b/i,
  /\bhelp\s+(?:me|us)\s+(?:with|build|fix|set\s*up)\b/i,
  // money on the table
  /\bbudget\b/i,
  /\bwilling\s+to\s+pay\b/i,
  /\bpaid\s+(?:gig|work|project|opportunity)\b/i,
  /\bhappy\s+to\s+pay\b/i,
  /\$\s?\d/,
  /\b\d+\s?(?:usd|eur|gbp|inr|k\b)/i,
  /\b(?:per|an)\s+hour\b/i,
  /\bquote\b/i,
  // outsourcing language
  /\b(?:outsource|freelancer|contractor|agency)\s+(?:for|to)\b/i,
  /\bdm\s+me\s+(?:your|with\s+your)\b/i,
  /\bsend\s+(?:me\s+)?your\s+(?:portfolio|rates|quote)\b/i,
  // problem statements a service would solve
  /\b(?:struggling|stuck|having\s+(?:trouble|issues))\s+with\b/i,
  /\b(?:is|are)\s+(?:broken|down|not\s+working)\b/i,
  /\bhow\s+(?:do|can)\s+(?:i|we)\s+(?:get|find|fix|hire)\b/i,
];

/** True when the post contains at least one phrase showing real demand. */
export function hasBuyingSignal(title: string, body = ''): boolean {
  const text = `${title}\n${body}`;
  return BUYING_SIGNALS.some((re) => re.test(text));
}

/**
 * A key that matches the same advert posted to several subreddits.
 *
 * Recruiters cross-post one role widely and reorder the title each time, so
 * exact-title matching does not catch it:
 *
 *   "[Hiring] Senior Backend Python Developer - Remote"
 *   "[Hiring] [Remote] [US] - Senior Backend Python Developer"
 *
 * Both collapse to the same key here. Without this, one job advert can fill
 * most of a requested lead list, which technically hits the number while
 * giving the user almost nothing to act on.
 */
export function titleFingerprint(title: string): string {
  const words = tokenize(title.replace(/\[[^\]]*\]/g, ' ')).map(stem);
  return [...new Set(words)].sort().join(' ');
}

/**
 * True when the post is someone advertising their own services.
 * A buyer marker wins, since posts sometimes carry both.
 */
export function looksLikeSeller(title: string, body = ''): boolean {
  const text = `${title}\n${body}`;
  if (BUYER_MARKERS.some((re) => re.test(text))) return false;
  return SELLER_MARKERS.some((re) => re.test(text));
}

export interface Candidate {
  intent: string;
  intent_confidence: number;
  relevance: number;
  /** Result of topicOverlap for this post. */
  overlap: number;
  /** From the role model: buyer | seller | advisor | other. */
  role?: string;
  role_confidence?: number;
  /** From looksLikeSeller — the deterministic flair check. */
  sellerFlair?: boolean;
  /** From looksLikeJobSeeker — someone seeking work, not buying. */
  jobSeeker?: boolean;
  /** From hasBuyingSignal — the post expresses actual demand. */
  buyingSignal?: boolean;
}

export interface GateConfig {
  /** Minimum semantic relevance for any lead. */
  minRelevance: number;
  /** Relevance high enough to accept on meaning alone, with no word overlap. */
  strongRelevance: number;
  /** Minimum share of query words that must appear in the post. */
  minOverlap: number;
  /** Allow advice_seeking as well as buying_intent. */
  allowAdviceSeeking: boolean;
  /** Minimum confidence from the intent model. */
  minIntentConfidence: number;
}

export const STRICT: GateConfig = {
  minRelevance: 0.32,
  strongRelevance: 0.55,
  minOverlap: 0.34, // at least a third of the query's meaningful words
  allowAdviceSeeking: true,
  minIntentConfidence: 0.45,
};

/** Why a candidate was rejected, for logging and for the stats block. */
export type Rejection =
  | 'intent'
  | 'confidence'
  | 'relevance'
  | 'offtopic'
  | 'seller'
  | 'jobseeker'
  | 'nodemand'
  | null;

export function rejectionReason(c: Candidate, cfg: GateConfig = STRICT): Rejection {
  // Competitors first. The app's premise is "you sell something, find
  // buyers", so a post advertising the same service is the opposite of a
  // lead, however well it matches the topic.
  if (c.sellerFlair) return 'seller';
  if (c.role === 'seller' && (c.role_confidence ?? 0) >= 0.5) return 'seller';

  // Then job hunters and careers chatter, which match the topic perfectly
  // and want to be hired rather than to hire.
  if (c.jobSeeker) return 'jobseeker';

  // Finally, the post must actually express demand. Passing every negative
  // check still leaves discussion threads, which are on-topic and harmless
  // and not leads.
  if (c.buyingSignal === false) return 'nodemand';

  const intentOk =
    c.intent === 'buying_intent' ||
    (cfg.allowAdviceSeeking && c.intent === 'advice_seeking');
  if (!intentOk) return 'intent';

  if (c.intent_confidence < cfg.minIntentConfidence) return 'confidence';
  if (c.relevance < cfg.minRelevance) return 'relevance';

  // Topic guard. -1 means the query had no meaningful words to check, so the
  // semantic score is all we have to go on.
  if (c.overlap >= 0) {
    const wordMatch = c.overlap >= cfg.minOverlap;
    const meaningMatch = c.relevance >= cfg.strongRelevance;
    if (!wordMatch && !meaningMatch) return 'offtopic';
  }

  return null;
}

export function qualifies(c: Candidate, cfg: GateConfig = STRICT): boolean {
  return rejectionReason(c, cfg) === null;
}

/**
 * Ranking score. Buying intent outranks advice seeking at equal relevance,
 * and word overlap breaks ties, so the most obviously on-topic leads sit at
 * the top of the list.
 */
export function leadScore(c: Candidate): number {
  const intentBonus = c.intent === 'buying_intent' ? 0.15 : 0;
  const overlapBonus = c.overlap > 0 ? c.overlap * 0.1 : 0;
  const buyerBonus = c.role === 'buyer' ? 0.1 : 0;
  return c.relevance + intentBonus + overlapBonus + buyerBonus;
}
