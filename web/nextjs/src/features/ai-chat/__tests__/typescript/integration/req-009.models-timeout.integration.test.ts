import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchChatModels } from "@/features/ai-chat/typescript/api/chatApi";

describe("REQ-009 model endpoint timeout semantics", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("returns deterministic timeout contract when fetch exceeds configured timeout", async () => {
    vi.useFakeTimers();

    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((_, init?: RequestInit) => {
        return new Promise<Response>((_resolve, reject) => {
          const signal = init?.signal as AbortSignal | undefined;
          signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        });
      });

    const pending = fetchChatModels({}, { timeoutMs: 25 });
    await vi.advanceTimersByTimeAsync(25);

    await expect(pending).resolves.toEqual({
      status: "error",
      error: {
        code: "MODEL_FETCH_TIMEOUT",
        retryable: true,
        message: "Model endpoint timed out.",
      },
    });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });
});
