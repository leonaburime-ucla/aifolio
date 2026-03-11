import { describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useChatLogic } from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import type {
  ChatDeps,
  ChatUiState,
} from "@/features/ai-chat/__types__/typescript/chat.types";
import { DEFAULT_CHAT_LOGIC_DEPS } from "@/features/ai-chat/__tests__/fixtures/chatLogicDeps.fixture";

describe("REQ-001/DR-001 empty input short-circuit", () => {
  it("does not mutate state or call API when normalized input is empty", async () => {
    const uiState: ChatUiState = {
      value: "    ",
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

    expect(deps.api.sendMessage).not.toHaveBeenCalled();
    expect(deps.actions.addInputToHistory).not.toHaveBeenCalled();
    expect(deps.actions.addMessage).not.toHaveBeenCalled();
    expect(deps.actions.setSending).not.toHaveBeenCalled();
    expect(deps.actions.resetHistoryCursor).not.toHaveBeenCalled();
    expect(uiState.resetValue).not.toHaveBeenCalled();
    expect(uiState.clearAttachments).not.toHaveBeenCalled();
  });
});
