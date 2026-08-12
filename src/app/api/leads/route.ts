// CRUD for saved leads. SQLite via Prisma.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

export const dynamic = 'force-dynamic';

export async function GET() {
  const leads = await prisma.lead.findMany({
    orderBy: { createdAt: 'desc' },
    take: 200,
  });
  return NextResponse.json({ leads });
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  if (!body.redditId || !body.title) {
    return NextResponse.json({ error: 'redditId and title required' }, { status: 400 });
  }
  try {
    const lead = await prisma.lead.upsert({
      where: { redditId: body.redditId },
      create: {
        redditId: body.redditId,
        title: (body.title || '').substring(0, 500),
        subreddit: body.subreddit || '',
        url: body.url || '',
        author: body.author || '',
        selftext: (body.selftext || '').substring(0, 5000),
        score: body.score || 0,
        numComments: body.numComments || 0,
        keyword: (body.keyword || '').substring(0, 200),
        relevanceScore: body.relevanceScore || 0,
        sentiment: body.sentiment || '',
        urgency: body.urgency || '',
        leadType: body.leadType || '',
        role: body.role || '',
        suggestedReply: (body.suggestedReply || '').substring(0, 2000),
        foundVia: 'manual',
      },
      update: {},
    });
    return NextResponse.json({ lead });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}

export async function PATCH(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  if (!body.id) return NextResponse.json({ error: 'id required' }, { status: 400 });
  const updates: any = {};
  if (typeof body.status === 'string') updates.status = body.status;
  await prisma.lead.update({ where: { id: body.id }, data: updates });
  return NextResponse.json({ ok: true });
}

export async function DELETE(req: NextRequest) {
  const id = Number(req.nextUrl.searchParams.get('id'));
  if (!id) return NextResponse.json({ error: 'id required' }, { status: 400 });
  await prisma.lead.delete({ where: { id } }).catch(() => {});
  return NextResponse.json({ ok: true });
}
