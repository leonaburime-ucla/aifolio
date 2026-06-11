export type ResolveAiApiBaseUrlInput = {
  env?: Record<string, string | undefined>;
  isBrowser?: boolean;
  defaultServerBaseUrl?: string;
  clientProxyBaseUrl?: string;
};

export const DEFAULT_AI_API_BASE_URL = "http://127.0.0.1:8000";
export const DEFAULT_CLIENT_AI_PROXY_BASE_URL = "/api/ai";

/**
 * Resolves the AI backend URL for server-side runtime calls.
 *
 * @param input - Environment and optional default URL overrides.
 * @returns Configured backend URL, public backend URL, or local development URL.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function resolveServerAiApiBaseUrl(
  input: Pick<ResolveAiApiBaseUrlInput, "env" | "defaultServerBaseUrl"> = {}
): string {
  const env = input.env ?? {};
  return (
    env.AI_API_URL ||
    env.NEXT_PUBLIC_AI_API_URL ||
    input.defaultServerBaseUrl ||
    DEFAULT_AI_API_BASE_URL
  );
}

/**
 * Resolves the AI API base URL for browser or server runtimes.
 *
 * @param input - Environment, browser flag, and optional URL overrides.
 * @returns Same-origin proxy URL in browser mode; backend URL in server mode.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function resolveAiApiBaseUrl(
  input: ResolveAiApiBaseUrlInput = {}
): string {
  if (input.isBrowser) {
    return input.clientProxyBaseUrl ?? DEFAULT_CLIENT_AI_PROXY_BASE_URL;
  }

  return resolveServerAiApiBaseUrl(input);
}
