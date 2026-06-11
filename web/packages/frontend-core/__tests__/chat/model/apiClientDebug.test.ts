import { describe, expect, it, vi } from "vitest";
import {
  sendChatMessageToEndpoint,
  fetchChatModels,
} from "../../../src/chat/model/apiClient";

function mockFetch(body: unknown, ok = true) {
  return vi.fn().mockResolvedValue({
    ok,
    status: ok ? 200 : 500,
    json: async () => body,
  });
}

const BASE_URL = "http://test-api";
function debugDeps(fetchImpl: typeof fetch) {
  return {
    runtimeDeps: {
      fetchImpl,
      resolveBaseUrl: () => BASE_URL,
      createAbortController: () => new AbortController(),
      setTimeoutImpl: setTimeout as typeof setTimeout,
      clearTimeoutImpl: clearTimeout as typeof clearTimeout,
      debug: true,
      logger: { warn: vi.fn() },
    },
  };
}

describe("sendChatMessageToEndpoint debug logging", () => {
  it("logs request info when debug is enabled", async () => {
    const fetchImpl = mockFetch({ result: JSON.stringify({ message: "ok" }) });
    const d = debugDeps(fetchImpl);
    await sendChatMessageToEndpoint(
      { endpoint: "/chat", value: "hi", model: "m" },
      d
    );
    expect(d.runtimeDeps.logger.warn).toHaveBeenCalledWith(
      "[ai-chat] request",
      expect.objectContaining({ endpoint: "/chat" })
    );
  });

  it("logs on non-ok response when debug is enabled", async () => {
    const fetchImpl = mockFetch({}, false);
    const d = debugDeps(fetchImpl);
    await expect(
      sendChatMessageToEndpoint({ endpoint: "/chat", value: "hi", model: "m" }, d)
    ).rejects.toMatchObject({ code: "CHAT_REQUEST_HTTP_ERROR" });
    expect(d.runtimeDeps.logger.warn).toHaveBeenCalledWith(
      "[ai-chat] request non-ok response",
      expect.objectContaining({ status: 500 })
    );
  });

  it("logs on parse error when debug is enabled", async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => { throw new Error("bad json"); },
    });
    const d = debugDeps(fetchImpl);
    await expect(
      sendChatMessageToEndpoint({ endpoint: "/chat", value: "hi", model: "m" }, d)
    ).rejects.toMatchObject({ code: "CHAT_RESPONSE_PARSE_ERROR" });
    expect(d.runtimeDeps.logger.warn).toHaveBeenCalledWith(
      "[ai-chat] request response parse failed",
      expect.objectContaining({ endpoint: "/chat" })
    );
  });

  it("logs invalid payload warning when normalized result is null", async () => {
    const fetchImpl = mockFetch({ result: null });
    const d = debugDeps(fetchImpl);
    const result = await sendChatMessageToEndpoint(
      { endpoint: "/chat", value: "hi", model: "m" },
      d
    );
    expect(result).toBeNull();
    expect(d.runtimeDeps.logger.warn).toHaveBeenCalledWith(
      "[ai-chat] request invalid payload",
      expect.objectContaining({ endpoint: "/chat" })
    );
  });
});

describe("fetchChatModels debug logging", () => {
  it("logs fetch start when debug is enabled", async () => {
    const fetchImpl = mockFetch({
      status: "ok",
      models: [{ id: "m", label: "M" }],
    });
    const d = debugDeps(fetchImpl);
    await fetchChatModels({}, d);
    expect(d.runtimeDeps.logger.warn).toHaveBeenCalledWith(
      "[ai-chat] fetch-models",
      expect.objectContaining({ url: expect.stringContaining("/llm/gemini-models") })
    );
  });

  it("logs non-ok response when debug is enabled", async () => {
    const fetchImpl = mockFetch({}, false);
    const d = debugDeps(fetchImpl);
    await fetchChatModels({}, d);
    expect(d.runtimeDeps.logger.warn).toHaveBeenCalledWith(
      "[ai-chat] fetch-models non-ok response",
      expect.objectContaining({ status: 500 })
    );
  });

  it("logs invalid payload when debug is enabled", async () => {
    const fetchImpl = mockFetch({ status: "error" });
    const d = debugDeps(fetchImpl);
    await fetchChatModels({}, d);
    expect(d.runtimeDeps.logger.warn).toHaveBeenCalledWith(
      "[ai-chat] fetch-models invalid payload",
      expect.objectContaining({ url: expect.stringContaining("/llm/gemini-models") })
    );
  });

  it("logs thrown error when debug is enabled", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error("network"));
    const d = debugDeps(fetchImpl);
    await fetchChatModels({}, d);
    expect(d.runtimeDeps.logger.warn).toHaveBeenCalledWith(
      "[ai-chat] fetch-models threw",
      expect.objectContaining({ error: expect.any(Error) })
    );
  });
});
