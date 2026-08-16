// Shared chip-filter option lists.
//
// These live outside the page files because Next.js App Router restricts
// `page.tsx` to a known export set (default, metadata, dynamic, …) and
// rejects any extra named export at type-check time.

import { POST_LIMIT_OPTIONS } from './limits';

export const TONE_OPTIONS = [
  { value: 'helpful', label: 'Helpful' },
  { value: 'professional', label: 'Professional' },
  { value: 'casual', label: 'Casual' },
  { value: 'empathetic', label: 'Empathetic' },
];

export const TIME_OPTIONS = [
  { value: 'day', label: 'Past day' },
  { value: 'week', label: 'Past week' },
  { value: 'month', label: 'Past month' },
  { value: 'year', label: 'Past year' },
];

/** Build a post-count option list, optionally capped for heavier features. */
export function buildLimitOptions(noun = 'posts', max = Infinity) {
  return POST_LIMIT_OPTIONS.filter((n) => n <= max).map((n) => ({
    value: String(n),
    label: `${n} ${noun}`,
    note: n >= 75 || (max <= 25 && n >= 25) ? 'slower' : undefined,
  }));
}
