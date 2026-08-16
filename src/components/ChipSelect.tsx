'use client';

// Chip-style dropdown used across every search panel: a compact pill button
// that opens a menu, with the active option marked in brand colour and a
// trailing check. Closes on outside click, Escape, or selection.

import { useEffect, useRef, useState } from 'react';
import { Icon } from './Icon';

export interface ChipOption {
  value: string;
  label: string;
  /** Optional note shown right-aligned in the menu (e.g. "slower"). */
  note?: string;
}

interface Props {
  icon?: 'calendar' | 'target' | 'message' | 'sliders' | 'compass' | 'layers';
  options: ChipOption[];
  value: string;
  onChange: (value: string) => void;
  /** Rendered instead of the matched option's label when set. */
  label?: string;
  disabled?: boolean;
  /** Dashed, muted styling — used for the inert "Advanced" chip. */
  muted?: boolean;
  ariaLabel?: string;
}

export function ChipSelect({
  icon,
  options,
  value,
  onChange,
  label,
  disabled,
  muted,
  ariaLabel,
}: Props) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);
  const active = options.find((o) => o.value === value);
  const text = label ?? active?.label ?? '';

  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  return (
    <div className="chip-wrap" ref={wrapRef}>
      <button
        type="button"
        className={`chip ${open ? 'open' : ''} ${muted ? 'muted' : ''}`}
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
      >
        {icon && <Icon name={icon} size={14} />}
        <span className="chip-text">{text}</span>
        <Icon name="chevron" size={13} className="chip-caret" />
      </button>

      {open && (
        <div className="chip-menu" role="listbox">
          {options.map((o) => {
            const selected = o.value === value;
            return (
              <button
                key={o.value}
                type="button"
                role="option"
                aria-selected={selected}
                className={`chip-menu-item ${selected ? 'selected' : ''}`}
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                }}
              >
                <span>{o.label}</span>
                <span className="chip-menu-right">
                  {o.note && <span className="chip-menu-note">{o.note}</span>}
                  {selected && <Icon name="check" size={13} />}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
