import { getServerAiApiBaseUrl } from "@/core/config/aiApi";

type ProxyRouteContext = {
  params: Promise<{
    path: string[];
  }>;
};

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "content-length",
  "host",
]);
const DEBUG_AI_PROXY = process.env.NODE_ENV === "development";

function buildProxyHeaders(request: Request): Headers {
  const headers = new Headers(request.headers);

  for (const header of HOP_BY_HOP_HEADERS) {
    headers.delete(header);
  }

  return headers;
}

async function proxyRequest(
  request: Request,
  context: ProxyRouteContext
): Promise<Response> {
  const { path } = await context.params;
  const incomingUrl = new URL(request.url);
  const targetUrl = new URL(
    `${getServerAiApiBaseUrl()}/${path.join("/")}${incomingUrl.search}`
  );
  const headers = buildProxyHeaders(request);
  const body =
    request.method === "GET" || request.method === "HEAD"
      ? undefined
      : await request.arrayBuffer();

  if (DEBUG_AI_PROXY) {
    console.warn("[api/ai proxy] forwarding", {
      method: request.method,
      incoming: `${incomingUrl.pathname}${incomingUrl.search}`,
      target: targetUrl.toString(),
    });
  }

  const upstream = await fetch(targetUrl, {
    method: request.method,
    headers,
    body,
    redirect: "manual",
    cache: "no-store",
  });

  if (DEBUG_AI_PROXY) {
    console.warn("[api/ai proxy] upstream response", {
      method: request.method,
      target: targetUrl.toString(),
      status: upstream.status,
    });
  }

  return new Response(
    request.method === "HEAD" ? null : await upstream.arrayBuffer(),
    {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: upstream.headers,
    }
  );
}

export async function GET(
  request: Request,
  context: ProxyRouteContext
): Promise<Response> {
  return proxyRequest(request, context);
}

export async function POST(
  request: Request,
  context: ProxyRouteContext
): Promise<Response> {
  return proxyRequest(request, context);
}

export async function PUT(
  request: Request,
  context: ProxyRouteContext
): Promise<Response> {
  return proxyRequest(request, context);
}

export async function PATCH(
  request: Request,
  context: ProxyRouteContext
): Promise<Response> {
  return proxyRequest(request, context);
}

export async function DELETE(
  request: Request,
  context: ProxyRouteContext
): Promise<Response> {
  return proxyRequest(request, context);
}

export async function OPTIONS(
  request: Request,
  context: ProxyRouteContext
): Promise<Response> {
  return proxyRequest(request, context);
}

export async function HEAD(
  request: Request,
  context: ProxyRouteContext
): Promise<Response> {
  return proxyRequest(request, context);
}
