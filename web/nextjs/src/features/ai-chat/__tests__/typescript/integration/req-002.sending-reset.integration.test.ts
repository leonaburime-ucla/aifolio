import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useChatLogic } from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import type {
  ChatAssistantPayload,
  ChatDeps,
  ChatUiState,
} from "@/features/ai-chat/__types__/typescript/chat.types";

function createDeps(
  sendMessage: ChatDeps["api"]["sendMessage"],
  uiState: ChatUiState
): ChatDeps {
  return {
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
      addMessage: vi.fn(),
      addInputToHistory: vi.fn(),
      moveHistoryCursor: vi.fn(() => ""),
      resetHistoryCursor: vi.fn(),
      setSending: vi.fn(),
      setModelOptions: vi.fn(),
      setSelectedModelId: vi.fn(),
      setModelsLoading: vi.fn(),
      addChartSpec: vi.fn(),
      onMessageReceived: vi.fn(),
    },
    api: {
      sendMessage,
      fetchModels: vi.fn(async () => null),
    },
  };
}

function createUiState(): ChatUiState {
  return {
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
}

describe("REQ-002 sending reset on all outcomes", () => {
  it("sets sending=false and clears attachments after success response", async () => {
    const uiState = createUiState();
    const deps = createDeps(
      vi.fn(async (): Promise<ChatAssistantPayload | null> => ({
        message: "assistant",
        chartSpec: null,
      })),
      uiState
    );

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    await act(async () => {
      await result.current.submit();
    });

    expect(deps.actions.setSending).toHaveBeenNthCalledWith(1, true);
    expect(deps.actions.setSending).toHaveBeenLastCalledWith(false);
    expect(uiState.clearAttachments).toHaveBeenCalledTimes(1);
  });

  it("sets sending=false and clears attachments after null response", async () => {
    const uiState = createUiState();
    const deps = createDeps(vi.fn(async () => null), uiState);

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    await act(async () => {
      await result.current.submit();
    });

    expect(deps.actions.setSending).toHaveBeenNthCalledWith(1, true);
    expect(deps.actions.setSending).toHaveBeenLastCalledWith(false);
    expect(uiState.clearAttachments).toHaveBeenCalledTimes(1);
  });

  it("sets sending=false and clears attachments after error", async () => {
    const uiState = createUiState();
    const deps = createDeps(
      vi.fn(async () => {
        throw new Error("network");
      }),
      uiState
    );

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    await expect(
      act(async () => {
        await result.current.submit();
      })
    ).rejects.toThrow("network");

    expect(deps.actions.setSending).toHaveBeenNthCalledWith(1, true);
    expect(deps.actions.setSending).toHaveBeenLastCalledWith(false);
    expect(uiState.clearAttachments).toHaveBeenCalledTimes(1);
  });
});
