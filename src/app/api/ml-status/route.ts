// Small proxy so the Nav can check ML server status without CORS issues.

import { NextResponse } from 'next/server';
import { isServerReachable } from '@/lib/mlClient';

export const dynamic = 'force-dynamic';

export async function GET() {
  const up = await isServerReachable();
  return NextResponse.json({ up }, { status: up ? 200 : 503 });
}
