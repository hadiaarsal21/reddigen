'use client';

// The hero search panel shared by Search, Deep Scan and Discover so every
// feature page presents the same shape: heading, one large query field,
// a row of chip filters, the submit button, then example queries.

import type { ReactNode } from 'react';
import { Icon } from './Icon';

interface Props {
  title: string;
  subtitle: ReactNode;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  loading: boolean;
  loadingLabel: string;
  submitLabel: string;
  /** ChipSelect elements. */
  filters?: ReactNode;
  /** Example queries; clicking one fills the field. */
  examples?: string[];
  /** Small print under the chip row. */
  hint?: ReactNode;
  maxLength?: number;
  minLength?: number;
}

export function SearchPanel({
  title,
  subtitle,
  placeholder,
  value,
  onChange,
  onSubmit,
  loading,
  loadingLabel,
  submitLabel,
  filters,
  examples,
  hint,
  maxLength = 200,
  minLength = 3,
}: Props) {
  const tooShort = value.trim().length < minLength;

  return (
    <form
      className="search-panel"
      onSubmit={(e) => {
        e.preventDefault();
        if (!loading && !tooShort) onSubmit();
      }}
    >
      <h2 className="search-panel-title">{title}</h2>
      <p className="search-panel-sub">{subtitle}</p>

      <div className="search-field">
        <Icon name="search" size={18} className="search-field-icon" />
        <input
          type="text"
          className="search-input"
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          maxLength={maxLength}
          aria-label={title}
        />
        {value && (
          <button
            type="button"
            className="search-clear"
            onClick={() => onChange('')}
            aria-label="Clear query"
          >
            <Icon name="x" size={14} />
          </button>
        )}
      </div>

      <div className="search-controls">
        <div className="search-chips">{filters}</div>
        <button
          type="submit"
          className="btn btn-primary search-submit"
          disabled={loading || tooShort}
          title={tooShort ? `Enter at least ${minLength} characters` : undefined}
        >
          {loading ? (
            <>
              <span className="spinner" />
              {loadingLabel}
            </>
          ) : (
            <>
              {submitLabel}
              <Icon name="arrow-right" size={15} />
            </>
          )}
        </button>
      </div>

      {hint && <div className="hint search-hint">{hint}</div>}

      {examples && examples.length > 0 && (
        <div className="try-row">
          <span className="try-label">
            <Icon name="sparkles" size={12} />
            TRY
          </span>
          {examples.map((ex) => (
            <button
              key={ex}
              type="button"
              className="try-chip"
              onClick={() => onChange(ex)}
              disabled={loading}
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </form>
  );
}
