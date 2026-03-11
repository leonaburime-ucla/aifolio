import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useChatLogic } from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import type { ChatDeps, ChatUiState } from "@/features/ai-chat/__types__/typescript/chat.types";
import { DEFAULT_CHAT_LOGIC_DEPS } from "@/features/ai-chat/__tests__/fixtures/chatLogicDeps.fixture";

describe("useChatLogic runtime deps", () => {
  it("uses injected now/createId for deterministic message ids", async () => {
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
        sendMessage: vi.fn(async () => ({ message: "assistant", chartSpec: null })),
        fetchModels: vi.fn(async () => null),
      },
      logic: DEFAULT_CHAT_LOGIC_DEPS,
    };

    const now = vi
      .fn<() => number>()
      .mockReturnValueOnce(100)
      .mockReturnValueOnce(200);
    const createId = vi.fn((timestamp: number) => `msg-${timestamp}`);

    const { result } = renderHook(() =>
      useChatLogic(uiState, deps, { now, createId })
    );

    await act(async () => {
      await result.current.submit();
    });

    expect(now).toHaveBeenCalledTimes(2);
    expect(createId).toHaveBeenNthCalledWith(1, 100);
    expect(createId).toHaveBeenNthCalledWith(2, 200);
    expect(deps.actions.addMessage).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        id: "msg-100",
        role: "user",
        content: "hello",
        createdAt: 100,
      })
    );
    expect(deps.actions.addMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        id: "msg-200",
        role: "assistant",
        content: "assistant",
        createdAt: 200,
      })
    );
  });
});
