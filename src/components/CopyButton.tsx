'use client';

// Copies text to the clipboard with visible confirmation, so the workflow is
// copy reply -> open the Reddit thread -> paste.

import { useEffect, useRef, useState } from 'react';
import { Icon } from './Icon';

interface Props {
  text: string;
  label?: string;
  copiedLabel?: string;
  className?: string;
  title?: string;
}

export function CopyButton({
  text,
  label = 'Copy reply',
  copiedLabel = 'Copied',
  className = 'btn btn-ghost',
  title,
}: Props) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  async function copy() {
    if (!text) return;
    let ok = false;
    try {
      // navigator.clipboard needs a secure context. localhost counts, but a
      // LAN address over plain http does not, which is exactly how this app
      // gets opened from a phone.
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        ok = true;
      }
    } catch {
      ok = false;
    }

    if (!ok) {
      // Fallback for non-secure contexts.
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.top = '-1000px';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand('copy');
        document.body.removeChild(ta);
      } catch {
        ok = false;
      }
    }

    setState(ok ? 'copied' : 'failed');
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setState('idle'), ok ? 1800 : 3000);
  }

  return (
    <button
      type="button"
      className={`${className} copy-btn ${state === 'copied' ? 'copied' : ''}`}
      onClick={copy}
      disabled={!text}
      title={title ?? (text ? 'Copy this reply to the clipboard' : 'No reply to copy')}
      aria-live="polite"
    >
      {state === 'copied' ? (
        <>
          <Icon name="check" size={13} />
          {copiedLabel}
        </>
      ) : state === 'failed' ? (
        <>
          <Icon name="alert" size={13} />
          Press Ctrl+C
        </>
      ) : (
        <>
          <Icon name="copy" size={13} />
          {label}
        </>
      )}
    </button>
  );
}
