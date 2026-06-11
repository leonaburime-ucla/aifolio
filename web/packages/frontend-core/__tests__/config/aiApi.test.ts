import { describe, expect, it } from "vitest";
import {
  DEFAULT_AI_API_BASE_URL,
  DEFAULT_CLIENT_AI_PROXY_BASE_URL,
  resolveAiApiBaseUrl,
  resolveServerAiApiBaseUrl,
} from "../../src/config/aiApi";

describe("aiApi config", () => {
  it("resolves server URL from private env before public env", () => {
    expect(
      resolveServerAiApiBaseUrl({
        env: {
          AI_API_URL: "http://private-ai-api",
          NEXT_PUBLIC_AI_API_URL: "http://public-ai-api",
        },
      })
    ).toBe("http://private-ai-api");
  });

  it("falls back to public env, override default, then package default", () => {
    expect(
      resolveServerAiApiBaseUrl({
        env: { NEXT_PUBLIC_AI_API_URL: "http://public-ai-api" },
      })
    ).toBe("http://public-ai-api");

    expect(
      resolveServerAiApiBaseUrl({
        env: {},
        defaultServerBaseUrl: "http://override-ai-api",
      })
    ).toBe("http://override-ai-api");

    expect(resolveServerAiApiBaseUrl()).toBe(DEFAULT_AI_API_BASE_URL);
  });

  it("uses a same-origin proxy URL for browser clients", () => {
    expect(
      resolveAiApiBaseUrl({
        env: { AI_API_URL: "http://private-ai-api" },
        isBrowser: true,
      })
    ).toBe(DEFAULT_CLIENT_AI_PROXY_BASE_URL);

    expect(
      resolveAiApiBaseUrl({
        isBrowser: true,
        clientProxyBaseUrl: "/custom-ai-proxy",
      })
    ).toBe("/custom-ai-proxy");
  });

  it("delegates non-browser resolution to server rules", () => {
    expect(
      resolveAiApiBaseUrl({
        env: { NEXT_PUBLIC_AI_API_URL: "http://public-ai-api" },
        isBrowser: false,
      })
    ).toBe("http://public-ai-api");
  });
});
