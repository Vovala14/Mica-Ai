// Blob access for the MICA site, tolerant of how the store was connected:
// - credentials: BLOB_READ_WRITE_TOKEN, a custom-prefixed *_READ_WRITE_TOKEN,
//   or Vercel OIDC with BLOB_STORE_ID (handled inside @vercel/blob);
// - access: a private store (preferred) or a public one. Public blobs get a
//   random suffix so their URLs cannot be guessed; listing still needs the token.
import { get, list, put } from '@vercel/blob';

export function token() {
  if (process.env.BLOB_READ_WRITE_TOKEN) return process.env.BLOB_READ_WRITE_TOKEN;
  const key = Object.keys(process.env).find((k) => k.endsWith('READ_WRITE_TOKEN'));
  return key ? process.env[key] : undefined;
}

const creds = () => (token() ? { token: token() } : {});
let access = process.env.BLOB_ACCESS || null;   // learned on first write

const isAccessMismatch = (e) => /access|public|private/i.test(String(e && e.message));

export async function save(path, body) {
  const opts = (a) => ({
    ...creds(), access: a, contentType: 'application/json',
    addRandomSuffix: a === 'public', allowOverwrite: a === 'private',
  });
  if (access) return put(path, body, opts(access));
  try {
    const r = await put(path, body, opts('private'));
    access = 'private';
    return r;
  } catch (e) {
    if (!isAccessMismatch(e)) throw e;
    const r = await put(path, body, opts('public'));
    access = 'public';
    return r;
  }
}

export async function listAll(prefix) {
  const blobs = [];
  let cursor;
  do {
    const page = await list({ ...creds(), prefix, cursor, limit: 1000 });
    blobs.push(...page.blobs);
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs;
}

export async function read(blob) {
  for (const a of access ? [access] : ['private', 'public']) {
    try {
      const res = await get(blob.url, { ...creds(), access: a });
      if (res) return await new Response(res.stream).text();
    } catch (e) {
      if (!isAccessMismatch(e)) throw e;
    }
  }
  return null;
}

export function describe() {
  return {
    token: process.env.BLOB_READ_WRITE_TOKEN ? 'BLOB_READ_WRITE_TOKEN'
      : (Object.keys(process.env).find((k) => k.endsWith('READ_WRITE_TOKEN')) || null),
    oidc_store: process.env.BLOB_STORE_ID ? 'BLOB_STORE_ID' : null,
    access,
  };
}
