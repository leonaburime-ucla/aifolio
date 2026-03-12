import { describe, expect, it, vi } from "vitest";

const fetchMock = vi.fn();

vi.mock("@/core/config/aiApi", () => ({
  getServerAiApiBaseUrl: () => "http://127.0.0.1:8000",
}));

vi.stubGlobal("fetch", fetchMock);

import { GET, POST } from "@/app/api/ai/[...path]/route";

describe("ai proxy route", () => {
  it("forwards GET requests to the configured backend", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    );

    const response = await GET(
      new Request("http://localhost:3000/api/ai/llm/gemini-models?x=1"),
      {
        params: Promise.resolve({
          path: ["llm", "gemini-models"],
        }),
      }
    );

    expect(fetchMock).toHaveBeenCalledWith(
      new URL("http://127.0.0.1:8000/llm/gemini-models?x=1"),
      expect.objectContaining({
        method: "GET",
        cache: "no-store",
      })
    );
    expect(response.status).toBe(200);
  });

  it("forwards POST bodies to the configured backend", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ status: "ok" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    );

    await POST(
      new Request("http://localhost:3000/api/ai/chat-research", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message: "ping" }),
      }),
      {
        params: Promise.resolve({
          path: ["chat-research"],
        }),
      }
    );

    expect(fetchMock).toHaveBeenLastCalledWith(
      new URL("http://127.0.0.1:8000/chat-research"),
      expect.objectContaining({
        method: "POST",
        body: expect.any(ArrayBuffer),
      })
    );
  });
});
