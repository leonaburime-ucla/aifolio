import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FALLBACK_CHAT_MODELS } from "@/features/ai-chat/typescript/logic/modelSelection.logic";
import { useChatLogic } from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import type { ChatDeps, ChatUiState } from "@/features/ai-chat/__types__/typescript/chat.types";
import { DEFAULT_CHAT_LOGIC_DEPS } from "@/features/ai-chat/__tests__/fixtures/chatLogicDeps.fixture";

describe("ERR-002 model fetch failure fallback", () => {
  it("applies deterministic fallback models when fetchModels throws", async () => {
    const uiState: ChatUiState = {
      value: "",
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
        screenFeedback: null,
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
        setScreenFeedback: vi.fn(),
        addChartSpec: vi.fn(),
        onMessageReceived: vi.fn(),
      },
      api: {
        sendMessage: vi.fn(async () => null),
        fetchModels: vi.fn(async () => {
          throw new Error("timeout");
        }),
      },
      logic: DEFAULT_CHAT_LOGIC_DEPS,
    };

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    await act(async () => {
      await result.current.refetchModels();
    });

    expect(deps.actions.setModelsLoading).toHaveBeenNthCalledWith(1, true);
    expect(deps.actions.setModelOptions).toHaveBeenCalledWith(FALLBACK_CHAT_MODELS);
    expect(deps.actions.setSelectedModelId).toHaveBeenCalledWith(
      FALLBACK_CHAT_MODELS[0]?.id ?? null
    );
    expect(deps.actions.setModelsLoading).toHaveBeenLastCalledWith(false);
  });
});
