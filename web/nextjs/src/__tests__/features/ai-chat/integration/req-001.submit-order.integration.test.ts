import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useChatLogic } from "@/features/ai-chat/react/hooks/useChat.hooks";
import type { ChatDeps } from "@aifolio/contracts/entities/chat";
import type { ChatUiState } from "@aifolio/contracts/entities/chat";
import { DEFAULT_CHAT_LOGIC_DEPS } from "@/__tests__/features/ai-chat/fixtures/chatLogicDeps.fixture";

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
    const addMessage = vi.fn();
    const addInputToHistory = vi.fn();
    const setSending = vi.fn();
    const sendMessage = vi.fn(async () => null);

    const deps: ChatDeps = {
      state: {
        messages: [],
        inputHistory: [],
        historyCursor: null,
        isSending: false,
        modelOptions: [],
        selectedModelId: null,
        isModelsLoading: false,
        screenFeedback: null,
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
        setScreenFeedback: vi.fn(),
        addChartSpec: vi.fn(),
        onMessageReceived: vi.fn(),
      },
      api: {
        sendMessage,
        fetchModels: vi.fn(async () => null),
      },
      logic: DEFAULT_CHAT_LOGIC_DEPS,
    };

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    await act(async () => {
      await result.current.submit();
    });

    const inputOrder = addInputToHistory.mock.invocationCallOrder[0];
    const messageOrder = addMessage.mock.invocationCallOrder[0];
    const sendingTrueOrder = setSending.mock.invocationCallOrder[0];
    const sendOrder = sendMessage.mock.invocationCallOrder[0];

    expect(inputOrder).toBeLessThan(messageOrder);
    expect(messageOrder).toBeLessThan(sendingTrueOrder);
    expect(sendingTrueOrder).toBeLessThan(sendOrder);
    expect(setSending).toHaveBeenNthCalledWith(1, true);
  });
});
