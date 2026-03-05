import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchChatModels } from "@/features/ai-chat/typescript/api/chatApi";

describe("ERR-005 timeout contract", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("returns deterministic retryable timeout contract shape", async () => {
    vi.useFakeTimers();

    vi.spyOn(globalThis, "fetch").mockImplementation((_, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal as AbortSignal | undefined;
        signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        });
      });
    });

    const pending = fetchChatModels({}, { timeoutMs: 5 });
    await vi.advanceTimersByTimeAsync(5);

    const result = (await pending) as unknown as {
      error?: { retryable: boolean; code: string };
    } | null;

    expect(result).not.toBeNull();
    expect(result?.error?.retryable).toBe(true);
    expect(result?.error?.code).toBe("MODEL_FETCH_TIMEOUT");
  });
});
