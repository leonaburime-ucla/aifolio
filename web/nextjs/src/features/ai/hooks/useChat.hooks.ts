import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  ChatMessage,
  ChatModelOption,
  ChatState,
  ChatStateActions,
  ChatHistoryDirection,
  ChatApiDeps,
  ChatUiState,
  ChatActions,
  ChatIntegration,
  ChatDeps,
} from "@/features/ai/types/chat.types";

const FALLBACK_MODELS: ChatModelOption[] = [
  { id: "gemini-3-flash-preview", label: "Gemini 3 Flash Preview" },
  { id: "gemini-3-pro-preview", label: "Gemini 3 Pro Preview" },
  { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
];


/**
 * Sets the fallback models in the store.
 *
 * @param actions - The state actions to mutate the store.
 * @param selectedModelId - The currently selected model ID (to determine if we need to set a default).
 */
export function setFallbackModels(
  actions: ChatStateActions,
  selectedModelId: string | null
) {
  actions.setModelOptions(FALLBACK_MODELS);
  if (!selectedModelId) {
    actions.setSelectedModelId(FALLBACK_MODELS[0]?.id ?? null);
  }
}

/**
 * Sets the fetched models in the store.
 *
 * @param actions - The state actions to mutate the store.
 * @param selectedModelId - The currently selected model ID.
 * @param result - The result from the API fetch containing models and current model.
 */
export function setFetchedModels(
  actions: ChatStateActions,
  selectedModelId: string | null,
  result: { currentModel: string | null; models: ChatModelOption[] }
) {
  actions.setModelOptions(result.models);
  if (!selectedModelId) {
    actions.setSelectedModelId(
      result.currentModel ?? result.models[0]?.id ?? null
    );
  }
}





/**
 * Hook 1: Manage local UI state for the chat bar.
 * @returns Local chat input state and setters.
 */
export function useChatUiState(): ChatUiState {
  const [value, setValue] = useState("");
  const [showTooltip, setShowTooltip] = useState(false);
  const [attachments, setAttachments] = useState<ChatUiState["attachments"]>(
    []
  );

  const resetValue = useCallback(() => setValue(""), []);
  const clearAttachments = useCallback(() => setAttachments([]), []);
  const addAttachments = useCallback((files: ChatUiState["attachments"]) => {
    setAttachments((current) => [...current, ...files]);
  }, []);
  const removeAttachment = useCallback((index: number) => {
    setAttachments((current) => current.filter((_, idx) => idx !== index));
  }, []);

  return {
    value,
    showTooltip,
    attachments,
    setShowTooltip,
    setValue,
    resetValue,
    addAttachments,
    clearAttachments,
    removeAttachment,
  };
}

/**
 * Hook 2: Encapsulate business logic and state mutations.
 * @param uiState - Local UI state for the input field.
 * @param deps - Injected state/actions/API dependencies.
 * @returns Chat actions used by UI components.
 */
export function useChatLogic(
  uiState: ChatUiState,
  deps: ChatDeps
): ChatActions {
  const draftValueRef = useRef("");
  const submit = useCallback(async () => {
    const trimmed = uiState.value.trim();
    if (!trimmed) return;

    deps.actions.addInputToHistory(trimmed);
    const history = [
      ...deps.state.messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      { role: "user" as const, content: trimmed, attachments: uiState.attachments },
    ].slice(-10);

    deps.actions.addMessage({
      id: String(Date.now()),
      role: "user",
      content: trimmed,
      createdAt: Date.now(),
    });

    uiState.resetValue();
    deps.actions.resetHistoryCursor();

    deps.actions.setSending(true);
    try {
      const assistantMessage = await deps.api.sendMessage(
        trimmed,
        deps.state.selectedModelId,
        history,
        uiState.attachments,
        deps.state.activeDatasetId ?? null
      );
      if (assistantMessage) {
        deps.actions.addMessage({
          id: String(Date.now()),
          role: "assistant",
          content: assistantMessage.message,
          chartSpec: null,
          createdAt: Date.now(),
        });
        deps.actions.onMessageReceived(assistantMessage);
      }
    } finally {
      deps.actions.setSending(false);
      uiState.clearAttachments();
    }
  }, [deps, uiState]);

  /**
   * Navigates the input history and updates the UI state with the historical value.
   *
   * @param direction - Whether to move 'up' (older) or 'down' (newer) in history.
   */
  const handleHistory = useCallback(
    (direction: ChatHistoryDirection) => {
      if (deps.state.inputHistory.length === 0) return;
      if (direction === "up" && deps.state.historyCursor === null) {
        draftValueRef.current = uiState.value;
      }
      const nextValue = deps.actions.moveHistoryCursor(direction);
      if (
        direction === "down" &&
        deps.state.historyCursor !== null &&
        nextValue === ""
      ) {
        uiState.setValue(draftValueRef.current);
        return;
      }
      uiState.setValue(nextValue);
    },
    [deps, uiState]
  );

  /**
   * Resets the history cursor to the initial state (null).
   */
  const resetHistoryCursor = useCallback(() => {
    deps.actions.resetHistoryCursor();
  }, [deps]);

  /**
   * Updates the selected model ID in the store.
   *
   * @param value - The ID of the model to select, or null to clear selection.
   */
  const setSelectedModelId = useCallback(
    (value: string | null) => {
      deps.actions.setSelectedModelId(value);
    },
    [deps]
  );

  /**
   * Fetches models from the API and updates the store.
   * Exposed for testing and manual re-fetching.
   */
  const refetchModels = useCallback(async () => {
    try {
      deps.actions.setModelsLoading(true);
      const result = await deps.api.fetchModels();
      if (!result) {
        setFallbackModels(deps.actions, deps.state.selectedModelId);
        return;
      }

      setFetchedModels(deps.actions, deps.state.selectedModelId, result);
    } catch (error) {
      setFallbackModels(deps.actions, deps.state.selectedModelId);
    } finally {
      deps.actions.setModelsLoading(false);
    }
  }, [deps]);

  return {
    submit,
    handleHistory,
    resetHistoryCursor,
    setSelectedModelId,
    refetchModels,
  };
}

/**
 * Hook 3: Integration layer that composes UI, state, and logic.
 * @param deps - Injected state/actions/API dependencies.
 * @returns Combined state + actions used by UI components.
 */
export function useChatIntegration(
  deps: ChatDeps
): ChatIntegration {
  const uiState = useChatUiState();
  const actions = useChatLogic(uiState, deps);
  const { state, actions: storeActions, api } = deps;

  useEffect(() => {
    if (state.modelOptions.length > 0 || state.isModelsLoading) return;

    actions.refetchModels();
  }, [
    state.isModelsLoading,
    state.modelOptions.length,
    actions,
  ]);


  return useMemo(
    () => ({
      ...deps.state,
      ...uiState,
      ...actions,
    }),
    [deps.state, uiState, actions]
  );
}
