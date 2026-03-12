import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  setFallbackModels,
  setFetchedModels,
  useChatIntegration,
  useChatLogic,
  useChatUiState,
} from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import {
  FALLBACK_CHAT_MODELS,
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
} from "@/features/ai-chat/typescript/logic/modelSelection.logic";
import {
  normalizeSubmissionValue,
  buildChatHistoryWindow,
  createUserChatMessage,
  createAssistantChatMessage,
  shouldRestoreDraftValue,
} from "@/features/ai-chat/typescript/logic/chatSubmission.logic";
import type { ChatDeps, ChatUiState } from "@/features/ai-chat/__types__/typescript/chat.types";

function createUiState(overrides: Partial<ChatUiState> = {}): ChatUiState {
  return {
    value: "",
    showTooltip: false,
    attachments: [],
    setShowTooltip: vi.fn(),
    setValue: vi.fn(),
    resetValue: vi.fn(),
    addAttachments: vi.fn(),
    clearAttachments: vi.fn(),
    removeAttachment: vi.fn(),
    ...overrides,
  };
}

function createDeps(overrides: Partial<ChatDeps> = {}): ChatDeps {
  return {
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
      fetchModels: vi.fn(async () => null),
    },
    logic: {
      normalizeSubmissionValue,
      buildChatHistoryWindow,
      createUserChatMessage,
      createAssistantChatMessage,
      shouldRestoreDraftValue,
      resolveFallbackModelSelection,
      resolveFetchedModelSelection,
    },
    ...overrides,
  };
}

describe("useChat hooks unit", () => {
  afterEach(() => {
    cleanup();
  });

  it("setFallbackModels and setFetchedModels respect selectedModel behavior", () => {
    const actions = {
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
    };

    setFallbackModels({ actions, selectedModelId: null, resolveFallbackModelSelection });
    expect(actions.setModelOptions).toHaveBeenCalledWith(FALLBACK_CHAT_MODELS);
    expect(actions.setSelectedModelId).toHaveBeenCalledWith(
      FALLBACK_CHAT_MODELS[0]?.id ?? null
    );

    actions.setSelectedModelId.mockClear();
    setFallbackModels({ actions, selectedModelId: "already-set", resolveFallbackModelSelection });
    expect(actions.setSelectedModelId).not.toHaveBeenCalled();

    setFetchedModels({
      actions,
      selectedModelId: null,
      result: {
        currentModel: "m2",
        models: [{ id: "m1", label: "Model 1" }],
      },
      resolveFetchedModelSelection,
    });
    expect(actions.setModelOptions).toHaveBeenCalledWith([
      { id: "m1", label: "Model 1" },
    ]);
    expect(actions.setSelectedModelId).toHaveBeenCalledWith("m2");
  });

  it("useChatUiState manages attachments and local value helpers", () => {
    const { result } = renderHook(() => useChatUiState());

    act(() => {
      result.current.setValue("hello");
      result.current.addAttachments([
        { name: "a.txt", type: "text/plain", size: 1, dataUrl: "data:a" },
        { name: "b.txt", type: "text/plain", size: 2, dataUrl: "data:b" },
      ]);
    });
    expect(result.current.value).toBe("hello");
    expect(result.current.attachments).toHaveLength(2);

    act(() => {
      result.current.removeAttachment(0);
    });
    expect(result.current.attachments).toEqual([
      { name: "b.txt", type: "text/plain", size: 2, dataUrl: "data:b" },
    ]);

    act(() => {
      result.current.resetValue();
      result.current.clearAttachments();
    });
    expect(result.current.value).toBe("");
    expect(result.current.attachments).toEqual([]);
  });

  it("handleHistory restores draft value when navigating back down from history", () => {
    const uiState = createUiState({ value: "draft" });
    const deps = createDeps({
      state: {
        messages: [],
        inputHistory: ["cmd-1"],
        historyCursor: null,
        isSending: false,
        modelOptions: [],
        selectedModelId: null,
        isModelsLoading: false,
        screenFeedback: null,
        activeDatasetId: null,
      },
    });
    const moveHistoryCursor = deps.actions.moveHistoryCursor as ReturnType<typeof vi.fn>;
    moveHistoryCursor.mockReturnValueOnce("cmd-1").mockReturnValueOnce("");

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    act(() => {
      result.current.handleHistory("up");
    });
    expect(uiState.setValue).toHaveBeenCalledWith("cmd-1");

    deps.state.historyCursor = 0;
    act(() => {
      result.current.handleHistory("down");
    });
    expect(uiState.setValue).toHaveBeenLastCalledWith("draft");
  });

  it("handleHistory no-ops when there is no history", () => {
    const uiState = createUiState({ value: "draft" });
    const deps = createDeps({
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
    });

    const { result } = renderHook(() => useChatLogic(uiState, deps));
    act(() => {
      result.current.handleHistory("up");
    });

    expect(deps.actions.moveHistoryCursor).not.toHaveBeenCalled();
    expect(uiState.setValue).not.toHaveBeenCalled();
  });

  it("refetchModels handles error result and successful result", async () => {
    const uiState = createUiState();
    const deps = createDeps({
      api: {
        sendMessage: vi.fn(async () => null),
        fetchModels: vi
          .fn()
          .mockResolvedValueOnce({
            status: "error" as const,
            error: {
              code: "MODEL_FETCH_FAILED" as const,
              retryable: true,
              message: "bad",
            },
          })
          .mockResolvedValueOnce({
            status: "ok" as const,
            currentModel: "m3",
            models: [{ id: "m3", label: "Model 3" }],
          }),
      },
    });

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    await act(async () => {
      await result.current.refetchModels();
    });
    expect(deps.actions.setModelOptions).toHaveBeenCalledWith(FALLBACK_CHAT_MODELS);

    await act(async () => {
      await result.current.refetchModels();
    });
    expect(deps.actions.setModelOptions).toHaveBeenCalledWith([
      { id: "m3", label: "Model 3" },
    ]);
    expect(deps.actions.setSelectedModelId).toHaveBeenCalledWith("m3");
  });

  it("exposes resetHistoryCursor and setSelectedModelId action passthroughs", () => {
    const uiState = createUiState();
    const deps = createDeps();

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    act(() => {
      result.current.resetHistoryCursor();
      result.current.setSelectedModelId("model-x");
      result.current.setScreenFeedback({
        kind: "error",
        code: "CHAT_REQUEST_FAILED",
        message: "failed",
      });
    });

    expect(deps.actions.resetHistoryCursor).toHaveBeenCalledTimes(1);
    expect(deps.actions.setSelectedModelId).toHaveBeenCalledWith("model-x");
    expect(deps.actions.setScreenFeedback).toHaveBeenCalledWith({
      kind: "error",
      code: "CHAT_REQUEST_FAILED",
      message: "failed",
    });
  });

  it("retries the last failed submission without adding a duplicate user message", async () => {
    const sendMessage = vi
      .fn()
      .mockRejectedValueOnce(new Error("network"))
      .mockResolvedValueOnce({
        message: "assistant reply",
        chartSpec: null,
      });
    const uiState = createUiState({
      value: "hello",
      attachments: [{ name: "file.txt", type: "text/plain", size: 1, dataUrl: "data:file" }],
    });
    const deps = createDeps({
      api: {
        sendMessage,
        fetchModels: vi.fn(async () => null),
      },
    });

    const { result } = renderHook(() => useChatLogic(uiState, deps));

    await act(async () => {
      await result.current.submit();
    });

    expect(deps.actions.addMessage).toHaveBeenCalledTimes(1);
    expect(deps.actions.setScreenFeedback).toHaveBeenCalledWith(
      expect.objectContaining({
        code: "CHAT_REQUEST_FAILED",
      })
    );

    await act(async () => {
      await result.current.retryLastSubmission();
    });

    expect(sendMessage).toHaveBeenCalledTimes(2);
    expect(deps.actions.addMessage).toHaveBeenCalledTimes(2);
    expect(deps.actions.onMessageReceived).toHaveBeenCalledWith({
      message: "assistant reply",
      chartSpec: null,
    });
  });

  it("useChatIntegration bootstraps model fetch when models are empty and not loading", async () => {
    const deps = createDeps({
      api: {
        sendMessage: vi.fn(async () => null),
        fetchModels: vi.fn(async () => ({
          status: "ok" as const,
          currentModel: "m1",
          models: [{ id: "m1", label: "Model 1" }],
        })),
      },
    });

    const { result } = renderHook(() => useChatIntegration(deps));

    expect(result.current.value).toBe("");
    expect(deps.api.fetchModels).toHaveBeenCalledTimes(1);
  });

  it("useChatIntegration skips bootstrap when models already loaded or loading", () => {
    const loadedDeps = createDeps({
      state: {
        messages: [],
        inputHistory: [],
        historyCursor: null,
        isSending: false,
        modelOptions: [{ id: "m1", label: "Model 1" }],
        selectedModelId: "m1",
        isModelsLoading: false,
        screenFeedback: null,
        activeDatasetId: null,
      },
    });
    renderHook(() => useChatIntegration(loadedDeps));
    expect(loadedDeps.api.fetchModels).not.toHaveBeenCalled();

    const loadingDeps = createDeps({
      state: {
        messages: [],
        inputHistory: [],
        historyCursor: null,
        isSending: false,
        modelOptions: [],
        selectedModelId: null,
        isModelsLoading: true,
        screenFeedback: null,
        activeDatasetId: null,
      },
    });
    renderHook(() => useChatIntegration(loadingDeps));
    expect(loadingDeps.api.fetchModels).not.toHaveBeenCalled();
  });

});
