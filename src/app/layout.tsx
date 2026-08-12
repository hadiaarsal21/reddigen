import type { Metadata } from 'next';
import { Nav } from '@/components/Nav';
import './globals.css';

export const metadata: Metadata = {
  title: 'ReddiGen — Local ML Build',
  description:
    'Local research build: five trained deep-learning models discover, classify, and rank buying-intent conversations on Reddit.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="rg-main">{children}</main>
      </body>
    </html>
  );
}
