import {
  resolveAiApiBaseUrl,
  resolveServerAiApiBaseUrl,
} from "@aifolio/frontend-core";

/**
 * Resolve the shared AI backend base URL for server-side calls.
 * Uses `AI_API_URL` when present, then `NEXT_PUBLIC_AI_API_URL`, otherwise local dev.
 */
export function getServerAiApiBaseUrl(): string {
  return resolveServerAiApiBaseUrl({ env: process.env });
}

/**
 * Resolve the shared AI backend base URL.
 *
 * Browser code uses the Next.js same-origin proxy to avoid CORS.
 * Server code talks directly to the configured backend.
 */
export function getAiApiBaseUrl(): string {
  return resolveAiApiBaseUrl({
    env: process.env,
    isBrowser: typeof window !== "undefined",
  });
}
