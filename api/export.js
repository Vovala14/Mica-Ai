// Owner-only download of every logged generation and rating as JSON Lines.
//   /api/export?key=ADMIN_KEY            -> mica-site-logs.jsonl
//   /api/export?key=ADMIN_KEY&summary=1  -> counts only
import { get, list } from '@vercel/blob';
import { timingSafeEqual } from 'node:crypto';

function authorized(request) {
  const key = process.env.ADMIN_KEY;
  const url = new URL(request.url);
  const given = url.searchParams.get('key') ||
    (request.headers.get('authorization') || '').replace(/^Bearer /, '');
  if (!key || !given || given.length !== key.length) return false;
  return timingSafeEqual(Buffer.from(given), Buffer.from(key));
}

async function listAll(prefix) {
  const blobs = [];
  let cursor;
  do {
    const page = await list({ prefix, cursor, limit: 1000 });
    blobs.push(...page.blobs);
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs;
}

async function read(blob) {
  const res = await get(blob.url, { access: 'private' });
  return res ? await new Response(res.stream).text() : null;
}

export async function GET(request) {
  if (!authorized(request)) return new Response('Unauthorized', { status: 401 });
  const [logs, feedback] = await Promise.all([listAll('logs/'), listAll('feedback/')]);
  if (new URL(request.url).searchParams.get('summary')) {
    return Response.json({ generations: logs.length, feedback: feedback.length });
  }
  const all = [...logs, ...feedback];
  const lines = [];
  for (let i = 0; i < all.length; i += 25) {
    const texts = await Promise.all(all.slice(i, i + 25).map(read));
    for (const t of texts) if (t) lines.push(t.replace(/\n/g, ' '));
  }
  return new Response(lines.join('\n') + (lines.length ? '\n' : ''), {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Content-Disposition': 'attachment; filename="mica-site-logs.jsonl"',
      'Cache-Control': 'no-store',
    },
  });
}
