// Owner-only download of every logged generation and rating as JSON Lines.
// The key goes in a header, never the URL, so it stays out of request logs
// and browser history:
//   curl -H "Authorization: Bearer ADMIN_KEY" https://SITE/api/export -o mica-site-logs.jsonl
//   .../api/export?summary=1  -> counts only
//   .../api/export?check=1    -> storage health check (write + list + read)
import { describe, listAll, read, save } from '../lib/blob.js';
import { timingSafeEqual } from 'node:crypto';

function authorized(request) {
  const key = process.env.ADMIN_KEY;
  const given = (request.headers.get('authorization') || '').replace(/^Bearer /, '');
  if (!key || !given || given.length !== key.length) return false;
  return timingSafeEqual(Buffer.from(given), Buffer.from(key));
}

export async function GET(request) {
  if (!authorized(request)) return new Response('Unauthorized', { status: 401 });
  const params = new URL(request.url).searchParams;
  if (params.get('check')) {
    const out = { env: { LOG_SECRET: !!process.env.LOG_SECRET, ADMIN_KEY: true } };
    try {
      const w = await save(`health/${Date.now()}.json`, JSON.stringify({ ok: true }));
      const found = await listAll('health/');
      out.write = 'ok';
      out.list = found.length;
      out.read = (await read(w)) ? 'ok' : 'failed';
    } catch (e) {
      out.error = String(e && e.message);
    }
    out.blob = describe();
    return Response.json(out, { headers: { 'Cache-Control': 'no-store' } });
  }
  const [logs, feedback] = await Promise.all([listAll('logs/'), listAll('feedback/')]);
  if (params.get('summary')) {
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
