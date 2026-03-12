import { afterEach, describe, expect, it, vi } from "vitest";

describe("aiApi config", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env.AI_API_URL;
    delete process.env.NEXT_PUBLIC_AI_API_URL;
  });

  it("uses same-origin proxy in browser contexts", async () => {
    vi.stubGlobal("window", {} as Window & typeof globalThis);
    const { getAiApiBaseUrl } = await import("@/core/config/aiApi");

    expect(getAiApiBaseUrl()).toBe("/api/ai");
  });

  it("uses server env when running on the server", async () => {
    process.env.AI_API_URL = "https://backend.example.com";
    const { getServerAiApiBaseUrl } = await import(
      "@/core/config/aiApi"
    );

    expect(getServerAiApiBaseUrl()).toBe("https://backend.example.com");
  });

  it("falls back to localhost on the server when no env is set", async () => {
    const { getServerAiApiBaseUrl } = await import("@/core/config/aiApi");

    expect(getServerAiApiBaseUrl()).toBe("http://127.0.0.1:8000");
  });
});
