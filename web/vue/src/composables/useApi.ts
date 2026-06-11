export function useApi() {
  const config = useRuntimeConfig();
  const baseUrl = config.public.apiBaseUrl as string;

  async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${baseUrl}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      throw new Error(`API ${res.status}: ${res.statusText}`);
    }
    return res.json();
  }

  return { baseUrl, fetchJson };
}
