const DEFAULT_AI_API_BASE_URL = "http://127.0.0.1:8000";
const CLIENT_AI_PROXY_BASE_URL = "/api/ai";

/**
 * Resolve the shared AI backend base URL for server-side calls.
 * Uses `AI_API_URL` when present, then `NEXT_PUBLIC_AI_API_URL`, otherwise local dev.
 */
export function getServerAiApiBaseUrl(): string {
  return (
    process.env.AI_API_URL ||
    process.env.NEXT_PUBLIC_AI_API_URL ||
    DEFAULT_AI_API_BASE_URL
  );
}

/**
 * Resolve the shared AI backend base URL.
 *
 * Browser code uses the Next.js same-origin proxy to avoid CORS.
 * Server code talks directly to the configured backend.
 */
export function getAiApiBaseUrl(): string {
  if (typeof window === "undefined") {
    return getServerAiApiBaseUrl();
  }

  return CLIENT_AI_PROXY_BASE_URL;
}
