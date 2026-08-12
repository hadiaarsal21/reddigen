'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { Icon } from './Icon';

interface Props {
  activeHref: string;
  crumb: string;
  children: React.ReactNode;
}

const NAV_GROUPS: Array<{
  label: string;
  items: Array<{
    href: string;
    label: string;
    icon: 'home' | 'search' | 'list' | 'compass' | 'layers' | 'brain' | 'gem';
  }>;
}> = [
  {
    label: 'Workspace',
    items: [
      { href: '/dashboard', label: 'Search', icon: 'search' },
      { href: '/leads', label: 'Leads', icon: 'list' },
    ],
  },
  {
    label: 'Deep mining',
    items: [
      { href: '/deep-scan', label: 'Deep Scan', icon: 'layers' },
      { href: '/discover', label: 'Discover Subreddits', icon: 'compass' },
    ],
  },
  {
    label: 'Research',
    items: [
      { href: '/models', label: 'ML Models', icon: 'brain' },
    ],
  },
];

export function DashboardShell({ activeHref, crumb, children }: Props) {
  const pathname = usePathname();
  const [mlUp, setMlUp] = useState<'checking' | 'up' | 'down'>('checking');
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    setDrawerOpen(false);
  }, [pathname]);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const res = await fetch('/api/ml-status', { cache: 'no-store' });
        if (cancelled) return;
        setMlUp(res.ok ? 'up' : 'down');
      } catch {
        if (!cancelled) setMlUp('down');
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
    <div className="app-root">
      <div
        className={`sidebar-backdrop ${drawerOpen ? 'show' : ''}`}
        onClick={() => setDrawerOpen(false)}
      />
      <aside className={`sidebar ${drawerOpen ? 'open' : ''}`}>
        <div className="sidebar-logo">
          <div className="logo-mark">R</div>
          <div className="logo-text">ReddiGen</div>
        </div>

        <div className="sidebar-new">
          <Link
            href="/dashboard"
            className="btn btn-primary"
            style={{ width: '100%', justifyContent: 'center' }}
          >
            <Icon name="sparkles" size={14} />
            New Search
          </Link>
        </div>

        <nav className="sidebar-nav">
          {NAV_GROUPS.map((group) => (
            <div key={group.label} className="nav-group">
              <div className="nav-group-label">{group.label}</div>
              {group.items.map((item) => {
                const isActive = activeHref === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`nav-item ${isActive ? 'active' : ''}`}
                  >
                    <Icon name={item.icon} size={16} />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="status-card">
            <div className="status-card-title">
              <span
                className={`status-dot ${
                  mlUp === 'up' ? 'ok' : mlUp === 'down' ? 'bad' : ''
                }`}
              />
              ML server {mlUp === 'up' ? 'online' : mlUp === 'down' ? 'offline' : '…'}
            </div>
            <div className="status-card-meta">
              {mlUp === 'up'
                ? 'localhost:8000 — 5 models ready'
                : mlUp === 'down'
                ? 'run: python ml/server.py'
                : 'checking…'}
            </div>
          </div>
        </div>
      </aside>

      <div>
        <header className="topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              className="mobile-hamburger"
              onClick={() => setDrawerOpen(true)}
              aria-label="Open navigation"
            >
              <Icon name="menu" size={18} />
            </button>
            <div className="crumb">
              <Link href="/dashboard" style={{ color: 'var(--text-secondary)' }}>
                Dashboard
              </Link>
              <span className="crumb-sep">/</span>
              <strong>{crumb}</strong>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <a
              href="https://github.com/projectsbyfarhan1107/reddigen-local"
              target="_blank"
              rel="noreferrer"
              className="btn btn-ghost"
              style={{ padding: '6px 12px', fontSize: 12 }}
            >
              GitHub <Icon name="external" size={12} />
            </a>
          </div>
        </header>
        <main className="main-content">{children}</main>
      </div>
    </div>
  );
}
