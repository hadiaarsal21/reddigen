'use client';

// Explains HOW a result set was produced, so widened matches are never passed
// off as exact ones. Three states:
//
//   info   results found, but the filter had to be relaxed to find them
//   empty  retrieval genuinely returned nothing, with the reason why
//   error  the request itself failed
//
// A relaxed result is a normal outcome, not a failure, and is styled as such.

import type { ReactNode } from 'react';
import { Icon } from './Icon';

type Kind = 'info' | 'empty' | 'error';

interface Props {
  kind: Kind;
  title: string;
  children?: ReactNode;
}

export function ResultNotice({ kind, title, children }: Props) {
  return (
    <div className={`result-notice ${kind}`}>
      <span className="result-notice-icon">
        <Icon name={kind === 'error' ? 'alert' : kind === 'empty' ? 'search' : 'sparkles'} size={15} />
      </span>
      <div>
        <strong>{title}</strong>
        {children && <div className="result-notice-body">{children}</div>}
      </div>
    </div>
  );
}

/** Human wording for each relaxation tier returned by /api/search. */
export const TIER_NOTES: Record<string, { title: string; body: string } | null> = {
  strict: null, // exact matches: nothing to explain
  intent_only: {
    title: 'Widened to intent matches',
    body:
      'No posts cleared both the intent and relevance bars, so these are posts showing buying or advice-seeking intent, ranked by how closely they match your query.',
  },
  relevance_only: {
    title: 'Widened to topical matches',
    body:
      'No posts showed clear buying intent, so these are the ones most related to your query. Expect more browsing and fewer ready buyers.',
  },
  best_effort: {
    title: 'Showing closest matches',
    body:
      'Nothing matched the filters, so these are the closest posts retrieved, ranked by relevance. Treat them as leads to skim rather than qualified ones.',
  },
};
