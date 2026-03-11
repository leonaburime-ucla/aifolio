import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useChatLogic } from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import type { ChatAssistantPayload, ChatDeps, ChatUiState } from "@/features/ai-chat/__types__/typescript/chat.types";
import { DEFAULT_CHAT_LOGIC_DEPS } from "@/features/ai-chat/__tests__/fixtures/chatLogicDeps.fixture";

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("REQ-007 abort/unmount behavior", () => {
  it("does not perform post-request state mutation after unmount", async () => {
    const deferred = createDeferred<ChatAssistantPayload | null>();

    const uiState: ChatUiState = {
      value: "hello",
      showTooltip: false,
      attachments: [],
      setShowTooltip: vi.fn(),
      setValue: vi.fn(),
      resetValue: vi.fn(),
      addAttachments: vi.fn(),
      clearAttachments: vi.fn(),
      removeAttachment: vi.fn(),
    };

    const addInputToHistory = vi.fn();
    const addMessage = vi.fn();
    const setSending = vi.fn();

    const deps: ChatDeps = {
      state: {
        messages: [],
        inputHistory: [],
        historyCursor: null,
        isSending: false,
        modelOptions: [],
        selectedModelId: null,
        isModelsLoading: false,
        activeDatasetId: null,
      },
      actions: {
        addMessage,
        addInputToHistory,
        moveHistoryCursor: vi.fn(() => ""),
        resetHistoryCursor: vi.fn(),
        setSending,
        setModelOptions: vi.fn(),
        setSelectedModelId: vi.fn(),
        setModelsLoading: vi.fn(),
        addChartSpec: vi.fn(),
        onMessageReceived: vi.fn(),
      },
      api: {
        sendMessage: vi.fn(() => deferred.promise),
        fetchModels: vi.fn(async () => null),
      },
      logic: DEFAULT_CHAT_LOGIC_DEPS,
    };

    const { result, unmount } = renderHook(() => useChatLogic(uiState, deps));

    let submitPromise: Promise<void>;
    await act(async () => {
      submitPromise = result.current.submit();
    });

    expect(setSending).toHaveBeenCalledWith(true);

    unmount();

    await act(async () => {
      deferred.resolve({ message: "ok", chartSpec: null });
      await submitPromise!;
    });

    // REQ-007: no post-unmount state mutation for the in-flight request.
    expect(setSending.mock.calls).toEqual([[true]]);
  });
});
