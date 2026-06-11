import { describe, expect, it, vi } from "vitest";
import {
  sendChatMessage,
  sendChatMessageDirect,
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
function deps(fetchImpl: typeof fetch) {
  return {
    runtimeDeps: {
      fetchImpl,
      resolveBaseUrl: () => BASE_URL,
      createAbortController: () => new AbortController(),
      setTimeoutImpl: setTimeout as typeof setTimeout,
      clearTimeoutImpl: clearTimeout as typeof clearTimeout,
      debug: false,
      logger: { warn: vi.fn() },
    },
  };
}

describe("sendChatMessageToEndpoint", () => {
  it("posts to the specified endpoint and returns normalized result", async () => {
    const fetchImpl = mockFetch({
      result: JSON.stringify({ message: "Hello from AI", chartSpec: null }),
    });
    const result = await sendChatMessageToEndpoint(
      { endpoint: "/chat", value: "hi", model: "gemini" },
      deps(fetchImpl)
    );
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/chat`);
    expect(init.method).toBe("POST");
    expect(result).not.toBeNull();
    expect(result!.message).toBe("Hello from AI");
  });

  it("throws ChatRequestError on non-ok response", async () => {
    const fetchImpl = mockFetch({}, false);
    await expect(
      sendChatMessageToEndpoint(
        { endpoint: "/chat", value: "hi", model: "gemini" },
        deps(fetchImpl)
      )
    ).rejects.toMatchObject({ code: "CHAT_REQUEST_HTTP_ERROR" });
  });

  it("throws CHAT_RESPONSE_PARSE_ERROR when json parsing fails", async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => { throw new Error("parse failed"); },
    });
    await expect(
      sendChatMessageToEndpoint(
        { endpoint: "/chat", value: "hi", model: "gemini" },
        deps(fetchImpl)
      )
    ).rejects.toMatchObject({ code: "CHAT_RESPONSE_PARSE_ERROR" });
  });

  it("includes dataset_id and history in body", async () => {
    const fetchImpl = mockFetch({ result: JSON.stringify({ message: "ok" }) });
    await sendChatMessageToEndpoint(
      { endpoint: "/chat", value: "hi", model: "m", history: [{ role: "user", content: "x" }] },
      { datasetId: "iris.csv", ...deps(fetchImpl) }
    );
    const body = JSON.parse(fetchImpl.mock.calls[0][1].body);
    expect(body.dataset_id).toBe("iris.csv");
    expect(body.messages).toEqual([{ role: "user", content: "x" }]);
  });
});

describe("sendChatMessage", () => {
  it("routes to /chat-research endpoint", async () => {
    const fetchImpl = mockFetch({ result: JSON.stringify({ message: "response" }) });
    await sendChatMessage({ value: "hello", model: "m" }, deps(fetchImpl));
    expect(fetchImpl.mock.calls[0][0]).toBe(`${BASE_URL}/chat-research`);
  });
});

describe("sendChatMessageDirect", () => {
  it("routes to /chat endpoint with null dataset", async () => {
    const fetchImpl = mockFetch({ result: JSON.stringify({ message: "response" }) });
    await sendChatMessageDirect({ value: "hello", model: "m" }, deps(fetchImpl));
    expect(fetchImpl.mock.calls[0][0]).toBe(`${BASE_URL}/chat`);
    const body = JSON.parse(fetchImpl.mock.calls[0][1].body);
    expect(body.dataset_id).toBeNull();
  });
});

describe("fetchChatModels", () => {
  it("returns model list on success", async () => {
    const fetchImpl = mockFetch({
      status: "ok",
      currentModel: "gemini-flash",
      models: [{ id: "gemini-flash", label: "Gemini Flash" }],
    });
    const result = await fetchChatModels({}, deps(fetchImpl));
    expect(result).toEqual({
      status: "ok",
      currentModel: "gemini-flash",
      models: [{ id: "gemini-flash", label: "Gemini Flash" }],
    });
  });

  it("returns null on non-ok response", async () => {
    const fetchImpl = mockFetch({}, false);
    const result = await fetchChatModels({}, deps(fetchImpl));
    expect(result).toBeNull();
  });

  it("returns error result on invalid payload", async () => {
    const fetchImpl = mockFetch({ status: "error" });
    const result = await fetchChatModels({}, deps(fetchImpl));
    expect(result).toMatchObject({ status: "error", error: { code: "MODEL_FETCH_FAILED" } });
  });

  it("returns timeout error when fetch aborts", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(Object.assign(new Error("aborted"), { name: "AbortError" }));
    const result = await fetchChatModels({}, deps(fetchImpl));
    expect(result).toMatchObject({
      status: "error",
      error: { code: "MODEL_FETCH_TIMEOUT", retryable: true },
    });
  });

  it("returns generic error on network failure", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error("network down"));
    const result = await fetchChatModels({}, deps(fetchImpl));
    expect(result).toMatchObject({
      status: "error",
      error: { code: "MODEL_FETCH_FAILED", retryable: true },
    });
  });
});
