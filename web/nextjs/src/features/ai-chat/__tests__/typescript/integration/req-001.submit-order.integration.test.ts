import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useChatLogic } from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import type { ChatDeps, ChatUiState } from "@/features/ai-chat/__types__/typescript/chat.types";
import { DEFAULT_CHAT_LOGIC_DEPS } from "@/features/ai-chat/__tests__/fixtures/chatLogicDeps.fixture";

describe("REQ-001/AC-001 submit ordering", () => {
  it("appends user input/message before sendMessage and sets sending=true before API call", async () => {
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
        sendMessage: vi.fn(async () => null),
        fetchModels: vi.fn(async () => null),
      },
      logic: DEFAULT_CHAT_LOGIC_DEPS,
    };

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    await act(async () => {
      await result.current.submit();
    });

    const inputOrder = deps.actions.addInputToHistory.mock.invocationCallOrder[0];
    const messageOrder = deps.actions.addMessage.mock.invocationCallOrder[0];
    const sendingTrueOrder = deps.actions.setSending.mock.invocationCallOrder[0];
    const sendOrder = deps.api.sendMessage.mock.invocationCallOrder[0];

    expect(inputOrder).toBeLessThan(messageOrder);
    expect(messageOrder).toBeLessThan(sendingTrueOrder);
    expect(sendingTrueOrder).toBeLessThan(sendOrder);
    expect(deps.actions.setSending).toHaveBeenNthCalledWith(1, true);
  });
});
