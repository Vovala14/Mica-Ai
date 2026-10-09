// Stores one generation or one rating in the private Blob store.
// Only outputs signed by api/flame.py or api/ember.py (LOG_SECRET) are accepted.
import { save } from '../lib/blob.js';
import { createHmac, timingSafeEqual } from 'node:crypto';

const MODELS = new Set(['flame-w-0.3.3', 'flame-w-0.3.2', 'flame-w-0.3.1', 'flame-w-memory-20261003', 'flame-w-b-20261001', 'flame-w-full40', 'ember-v02a', 'ember-v0.3a', 'ember-v0.3.1']);
const str = (v, max) => (typeof v === 'string' ? v.slice(0, max) : '');

function validSig(r) {
  const key = process.env.LOG_SECRET;
  if (!key || typeof r.sig !== 'string' || r.sig.length !== 64) return false;
  const msg = [r.id, r.model, r.mode, r.prompt, r.output].join('\n');
  const want = createHmac('sha256', key).update(msg, 'utf8').digest();
  return timingSafeEqual(want, Buffer.from(r.sig, 'hex'));
}

export async function POST(request) {
  let r;
  try {
    r = await request.json();
  } catch {
    return Response.json({ error: 'Bad request.' }, { status: 400 });
  }
  if (!r || !MODELS.has(r.model) || !/^[0-9a-f]{32}$/.test(r.id || '') ||
      typeof r.prompt !== 'string' || typeof r.output !== 'string' || !validSig(r)) {
    return Response.json({ error: 'Not a signed MICA output.' }, { status: 403 });
  }
  const now = new Date();
  const promptId = typeof r.prompt_id === 'string' && /^[a-z][a-z0-9-]{0,23}$/.test(r.prompt_id) ? r.prompt_id : '';
  const base = {
    id: r.id, time: now.toISOString(), model: r.model, mode: str(r.mode, 40),
    prompt: r.prompt, output: r.output, session: str(r.session, 40),
    ...(promptId ? { prompt_id: promptId } : {}),
  };
  let path, record;
  if (r.kind === 'feedback') {
    const rating = ['up', 'down'].includes(r.rating) ? r.rating : null;
    const correction = str(r.correction, 400).trim();
    if (!rating && !correction) return Response.json({ error: 'Empty feedback.' }, { status: 400 });
    record = { kind: 'feedback', ...base, rating, correction };
    path = `feedback/${now.toISOString().slice(0, 10)}/${r.id}-${now.getTime()}.json`;
  } else {
    record = {
      kind: 'generation', ...base, ms: Number(r.ms) || null,
      settings: typeof r.settings === 'object' && r.settings ? r.settings : {},
      model_sha256: str(r.model_sha256, 64),
    };
    path = `logs/${now.toISOString().slice(0, 10)}/${r.id}.json`;
  }
  try {
    await save(path, JSON.stringify(record));
  } catch (e) {
    console.error('blob save failed:', e && e.message);
    return Response.json({ error: 'Storage is not set up.' }, { status: 503 });
  }
  return Response.json({ ok: true });
}
