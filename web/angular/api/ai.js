export default async function handler(req, res) {
  const path = String(req.query.path || '').replace(/^\/+/, '');
  const baseUrl = (process.env.ANGULAR_PUBLIC_AI_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
  const incomingUrl = new URL(req.url || '/', 'https://aifolio.local');
  const targetUrl = new URL(path ? `${baseUrl}/${path}` : baseUrl);

  incomingUrl.searchParams.forEach((value, key) => {
    if (key !== 'path') targetUrl.searchParams.append(key, value);
  });

  const headers = new Headers();
  if (req.headers['content-type']) headers.set('content-type', req.headers['content-type']);
  if (req.headers.authorization) headers.set('authorization', req.headers.authorization);

  const body =
    req.method === 'GET' || req.method === 'HEAD'
      ? undefined
      : await readBody(req);

  const upstream = await fetch(targetUrl, {
    method: req.method,
    headers,
    body,
  });

  res.status(upstream.status);
  upstream.headers.forEach((value, key) => {
    if (key.toLowerCase() !== 'transfer-encoding') res.setHeader(key, value);
  });

  if (req.method === 'HEAD') {
    res.end();
    return;
  }

  res.send(Buffer.from(await upstream.arrayBuffer()));
}

async function readBody(req) {
  if (req.body !== undefined) {
    if (Buffer.isBuffer(req.body) || typeof req.body === 'string') return req.body;
    return JSON.stringify(req.body);
  }

  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}
