import { describe, expect, it, vi } from "vitest";
import {
  fetchChatModels,
  sendChatMessage,
  sendChatMessageDirect,
} from "@/features/ai-chat/typescript/api/chatApi";

describe("chatApi", () => {
  it("sends research chat request and normalizes object payload", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        status: "ok",
        result: { message: "assistant", chartSpec: null },
      }),
    }));

    const result = await sendChatMessage(
      {
        value: "hello",
        model: "m1",
        history: [{ role: "user", content: "hello" }],
        attachments: [],
      },
      {
        datasetId: "d1",
        runtimeDeps: {
          fetchImpl: fetchImpl as unknown as typeof fetch,
          resolveBaseUrl: () => "http://ai-api",
        },
      }
    );

    expect(fetchImpl).toHaveBeenCalledWith("http://ai-api/chat-research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hello",
        attachments: [],
        model: "m1",
        messages: [{ role: "user", content: "hello" }],
        dataset_id: "d1",
      }),
    });
    expect(result).toEqual({ message: "assistant", chartSpec: null });
  });

  it("sends direct chat request and forces null dataset", async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        status: "ok",
        result: "plain text response",
      }),
    }));

    const result = await sendChatMessageDirect(
      {
        value: "hello",
        model: null,
        history: [],
        attachments: [],
      },
      {
        runtimeDeps: {
          fetchImpl: fetchImpl as unknown as typeof fetch,
          resolveBaseUrl: () => "http://ai-api",
        },
      }
    );

    expect(fetchImpl).toHaveBeenCalledWith("http://ai-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hello",
        attachments: [],
        model: null,
        messages: [],
        dataset_id: null,
      }),
    });
    expect(result).toEqual({ message: "plain text response", chartSpec: null });
  });

  it("throws on non-ok chat response and unreadable JSON, and returns null for invalid payload", async () => {
    const nonOkFetch = vi.fn(async () => ({ ok: false, status: 503 }));
    await expect(
      sendChatMessage(
        {
          value: "x",
          model: null,
          history: [],
          attachments: [],
        },
        {
          runtimeDeps: {
            fetchImpl: nonOkFetch as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
          },
        }
      )
    ).rejects.toMatchObject({
      code: "CHAT_REQUEST_HTTP_ERROR",
      status: 503,
    });

    const invalidPayloadFetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ status: "ok", result: [{ type: "text" }] }),
    }));
    const invalidResult = await sendChatMessage(
      {
        value: "x",
        model: null,
        history: [],
        attachments: [],
      },
      {
        runtimeDeps: {
          fetchImpl: invalidPayloadFetch as unknown as typeof fetch,
          resolveBaseUrl: () => "http://ai-api",
        },
      }
    );
    expect(invalidResult).toBeNull();

    const invalidJsonFetch = vi.fn(async () => ({
      ok: true,
      json: async () => {
        throw new Error("invalid json");
      },
    }));
    await expect(
      sendChatMessage(
        {
          value: "x",
          model: null,
          history: [],
          attachments: undefined,
        },
        {
          runtimeDeps: {
            fetchImpl: invalidJsonFetch as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
          },
        }
      )
    ).rejects.toMatchObject({
      code: "CHAT_RESPONSE_PARSE_ERROR",
    });
  });

  it("fetches models success path and invalid payload path", async () => {
    const successFetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        status: "ok",
        currentModel: "m2",
        models: [{ id: "m1", label: "Model 1" }],
      }),
    }));
    await expect(
      fetchChatModels(
        {},
        {
          runtimeDeps: {
            fetchImpl: successFetch as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
          },
        }
      )
    ).resolves.toEqual({
      status: "ok",
      currentModel: "m2",
      models: [{ id: "m1", label: "Model 1" }],
    });

    const undefinedCurrentModelFetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        status: "ok",
        models: [{ id: "m1", label: "Model 1" }],
      }),
    }));
    await expect(
      fetchChatModels(
        {},
        {
          runtimeDeps: {
            fetchImpl: undefinedCurrentModelFetch as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
          },
        }
      )
    ).resolves.toEqual({
      status: "ok",
      currentModel: null,
      models: [{ id: "m1", label: "Model 1" }],
    });

    const nonOkFetch = vi.fn(async () => ({ ok: false }));
    await expect(
      fetchChatModels(
        {},
        {
          runtimeDeps: {
            fetchImpl: nonOkFetch as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
          },
        }
      )
    ).resolves.toBeNull();

    const invalidFetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ status: "ok", models: undefined }),
    }));
    await expect(
      fetchChatModels(
        {},
        {
          runtimeDeps: {
            fetchImpl: invalidFetch as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
          },
        }
      )
    ).resolves.toEqual({
      status: "error",
      error: {
        code: "MODEL_FETCH_FAILED",
        retryable: true,
        message: "Model endpoint returned an invalid payload.",
      },
    });
  });

  it("returns deterministic failure contract on fetch rejection", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("boom");
    });
    await expect(
      fetchChatModels(
        {},
        {
          runtimeDeps: {
            fetchImpl: fetchImpl as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
          },
        }
      )
    ).resolves.toEqual({
      status: "error",
      error: {
        code: "MODEL_FETCH_FAILED",
        retryable: true,
        message: "Model endpoint request failed.",
      },
    });
  });

  it("treats non-abort DOMException as model fetch failure", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new DOMException("network down", "NetworkError");
    });

    await expect(
      fetchChatModels(
        {},
        {
          runtimeDeps: {
            fetchImpl: fetchImpl as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
          },
        }
      )
    ).resolves.toEqual({
      status: "error",
      error: {
        code: "MODEL_FETCH_FAILED",
        retryable: true,
        message: "Model endpoint request failed.",
      },
    });
  });

  it("always clears timeout via runtime deps finally path", async () => {
    const setTimeoutImpl = vi.fn((callback: () => void) => {
      callback();
      return 777 as unknown as ReturnType<typeof setTimeout>;
    });
    const clearTimeoutImpl = vi.fn();
    const fetchImpl = vi.fn(async () => {
      throw new DOMException("aborted", "AbortError");
    });

    await fetchChatModels(
      {},
      {
        timeoutMs: 1,
        runtimeDeps: {
          fetchImpl: fetchImpl as unknown as typeof fetch,
          resolveBaseUrl: () => "http://ai-api",
          setTimeoutImpl,
          clearTimeoutImpl,
        },
      }
    );

    expect(setTimeoutImpl).toHaveBeenCalled();
    expect(clearTimeoutImpl).toHaveBeenCalledWith(777);
  });

  it("clears timeout on successful model fetch when runtime deps are injected", async () => {
    const setTimeoutImpl = vi.fn(() => 123 as unknown as ReturnType<typeof setTimeout>);
    const clearTimeoutImpl = vi.fn();
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        status: "ok",
        currentModel: "m1",
        models: [{ id: "m1", label: "Model 1" }],
      }),
    }));

    await expect(
      fetchChatModels(
        {},
        {
          runtimeDeps: {
            fetchImpl: fetchImpl as unknown as typeof fetch,
            resolveBaseUrl: () => "http://ai-api",
            setTimeoutImpl,
            clearTimeoutImpl,
          },
        }
      )
    ).resolves.toEqual({
      status: "ok",
      currentModel: "m1",
      models: [{ id: "m1", label: "Model 1" }],
    });

    expect(clearTimeoutImpl).toHaveBeenCalledWith(123);
  });
});
