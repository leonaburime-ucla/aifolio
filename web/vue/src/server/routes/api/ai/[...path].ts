export default defineEventHandler(async (event) => {
  const path = event.context.params?.path ?? "";
  const config = useRuntimeConfig();
  const baseUrl = config.public.apiBaseUrl || "http://127.0.0.1:8000";
  const targetUrl = `${baseUrl}/${path}`;

  const method = event.method;
  const headers = new Headers();
  const incomingHeaders = getHeaders(event);
  if (incomingHeaders["content-type"]) {
    headers.set("content-type", incomingHeaders["content-type"]);
  }
  if (incomingHeaders["authorization"]) {
    headers.set("authorization", incomingHeaders["authorization"]);
  }

  const body =
    method === "GET" || method === "HEAD"
      ? undefined
      : await readRawBody(event);

  const upstream = await fetch(targetUrl, {
    method,
    headers,
    body,
  });

  setResponseStatus(event, upstream.status);
  const responseHeaders = Object.fromEntries(upstream.headers.entries());
  for (const [key, value] of Object.entries(responseHeaders)) {
    if (key.toLowerCase() !== "transfer-encoding") {
      setResponseHeader(event, key, value);
    }
  }

  return upstream.body;
});
