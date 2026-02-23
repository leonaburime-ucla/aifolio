const DEFAULT_AI_API_BASE_URL = "http://127.0.0.1:8000";

/**
 * Resolve the shared AI backend base URL.
 * Uses NEXT_PUBLIC_AI_API_URL when present, otherwise falls back to local dev.
 */
export function getAiApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_AI_API_URL || DEFAULT_AI_API_BASE_URL;
}
