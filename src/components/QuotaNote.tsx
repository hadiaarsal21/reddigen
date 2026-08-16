'use client';

// Small line under each search panel showing the per-minute budget left.
// Populated from the `quota` block the API returns; before the first request
// it just explains the limit exists.

import { Icon } from './Icon';

interface Props {
  quota: { remaining: number; limit: number } | null;
  /** Plural noun for the action, e.g. "searches". */
  unit: string;
  note?: string;
}

export function QuotaNote({ quota, unit, note }: Props) {
  return (
    <span className="quota-note">
      <Icon name="alert" size={13} />
      {quota ? (
        <span>
          <strong>
            {quota.remaining} of {quota.limit}
          </strong>{' '}
          {unit} left this minute.
        </span>
      ) : (
        <span>
          Limited to <strong>{unit}</strong> at a modest rate to stay inside
          Reddit&apos;s limits.
        </span>
      )}
      {note && <span> {note}</span>}
    </span>
  );
}
