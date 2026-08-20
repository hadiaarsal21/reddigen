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

/**
 * Search no longer relaxes its filter to fill a list. It keeps searching from
 * more angles until it has the requested number of genuinely qualifying
 * leads, and reports a shortfall rather than padding with weaker matches.
 * The dashboard builds that message from the stats block.
 */
