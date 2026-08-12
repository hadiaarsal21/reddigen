'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

const LINKS = [
  { href: '/', label: 'Home' },
  { href: '/dashboard', label: 'Dashboard' },
  { href: '/leads', label: 'Leads' },
  { href: '/models', label: 'ML Models' },
];

export function Nav() {
  const pathname = usePathname();
  const [mlStatus, setMlStatus] = useState<'checking' | 'up' | 'down'>('checking');

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const res = await fetch('/api/ml-status', { cache: 'no-store' });
        if (cancelled) return;
        setMlStatus(res.ok ? 'up' : 'down');
      } catch {
        if (!cancelled) setMlStatus('down');
      }
    }
    check();
    const iv = setInterval(check, 15_000);
    return () => {
      cancelled = true;
      clearInterval(iv);
    };
  }, []);

  return (
    <header className="rg-nav">
      <div className="rg-nav-inner">
        <Link href="/" className="rg-brand">ReddiGen</Link>
        <nav className="rg-nav-links">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={pathname === l.href ? 'active' : ''}
            >
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="rg-nav-status" title="Local FastAPI ML service">
          <span className={`rg-status-dot ${mlStatus === 'up' ? 'ok' : mlStatus === 'down' ? 'bad' : ''}`} />
          ML {mlStatus === 'up' ? 'online' : mlStatus === 'down' ? 'offline' : '…'}
        </div>
      </div>
    </header>
  );
}
