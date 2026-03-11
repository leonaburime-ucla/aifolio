import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  ChatActions,
  ChatDeps,
  ChatHistoryDirection,
  ChatIntegration,
  ChatModelOption,
  ChatStateActions,
  ChatUiState,
} from "@/features/ai-chat/__types__/typescript/chat.types";

const DEBUG_EFFECTS = process.env.NEXT_PUBLIC_DEBUG_EFFECTS === "1";

function getDebugPath(): string {
  return globalThis.location?.pathname ?? "";
}

export type ChatLogicRuntimeDeps = {
  now?: () => number;
  createId?: (timestamp: number) => string;
};

/**
 * Apply deterministic fallback model selection into state actions.
 *
 * @param input - Required fallback application inputs.
 */
export function setFallbackModels(input: {
  actions: ChatStateActions;
  selectedModelId: string | null;
  resolveFallbackModelSelection: ChatDeps["logic"]["resolveFallbackModelSelection"];
}): void {
  const selection = input.resolveFallbackModelSelection({
    selectedModelId: input.selectedModelId,
  });

  input.actions.setModelOptions(selection.modelOptions);
  if (!input.selectedModelId) {
    input.actions.setSelectedModelId(selection.selectedModelId);
  }
}

/**
 * Apply fetched model selection into state actions.
 *
 * @param input - Required fetched-model application inputs.
 */
export function setFetchedModels(input: {
  actions: ChatStateActions;
  selectedModelId: string | null;
  result: { currentModel: string | null; models: ChatModelOption[] };
  resolveFetchedModelSelection: ChatDeps["logic"]["resolveFetchedModelSelection"];
}): void {
  const selection = input.resolveFetchedModelSelection({
    selectedModelId: input.selectedModelId,
    result: input.result,
  });

  input.actions.setModelOptions(selection.modelOptions);
  if (!input.selectedModelId) {
    input.actions.setSelectedModelId(selection.selectedModelId);
  }
}

/**
 * Local input-layer state for chat UI only.
 *
 * @returns UI state and local mutators for input/attachments.
 */
export function useChatUiState(): ChatUiState {
  const [value, setValue] = useState("");
  const [showTooltip, setShowTooltip] = useState(false);
  const [attachments, setAttachments] = useState<ChatUiState["attachments"]>(
    []
  );

  /** Clears the input value after successful submit. */
  const resetValue = useCallback((): void => setValue(""), []);

  /** Removes all staged attachments after request completion. */
  const clearAttachments = useCallback((): void => setAttachments([]), []);

  /**
   * Append files to the current staged attachment list.
   *
   * @param files - Required attachment list to append.
   * @returns void
   */
  const addAttachments = useCallback((files: ChatUiState["attachments"]): void => {
    setAttachments((current) => [...current, ...files]);
  }, []);

  /**
   * Remove one staged attachment by positional index.
   *
   * @param index - Required attachment index.
   * @returns void
   */
  const removeAttachment = useCallback((index: number): void => {
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
 * Business-facing chat actions that coordinate UI state, store actions, and API calls.
 *
 * All logic functions (normalization, history, message creation) are injected
 * through `deps.logic` per Orc-BASH convention — no direct logic imports.
 *
 * @param uiState - Required local UI state object.
 * @param deps - Required injected state/actions/API/logic dependencies.
 * @returns Chat actions consumed by views.
 */
export function useChatLogic(
  uiState: ChatUiState,
  deps: ChatDeps,
  runtimeDeps: ChatLogicRuntimeDeps = {}
): ChatActions {
  const now = runtimeDeps.now ?? Date.now;
  const createId = runtimeDeps.createId ?? ((timestamp: number) => String(timestamp));
  /**
   * Stores the in-progress draft while navigating history.
   * Used to restore the draft when history traversal returns to latest entry.
   */
  const draftValueRef = useRef("");
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  /**
   * Main submit flow for chat requests.
   *
   * @remarks
   * Pipeline: normalize input -> append user message/history -> set sending ->
   * send request -> append assistant message -> finalize flags/attachments.
   *
   * @returns Promise that resolves when the submit lifecycle completes.
   */
  const submit = useCallback(async (): Promise<void> => {
    const trimmed = deps.logic.normalizeSubmissionValue({ value: uiState.value });
    if (!trimmed) return;

    deps.actions.addInputToHistory(trimmed);
    const history = deps.logic.buildChatHistoryWindow({
      messages: deps.state.messages,
      userContent: trimmed,
      attachments: uiState.attachments,
    });

    const userTimestamp = now();
    deps.actions.addMessage({
      ...deps.logic.createUserChatMessage({
        id: createId(userTimestamp),
        content: trimmed,
        createdAt: userTimestamp,
      }),
    });

    uiState.resetValue();
    deps.actions.resetHistoryCursor();

    deps.actions.setSending(true);
    try {
      const assistantMessage = await deps.api.sendMessage(
        {
          value: trimmed,
          model: deps.state.selectedModelId,
          history,
          attachments: uiState.attachments,
        },
        {
          datasetId: deps.state.activeDatasetId ?? null,
        }
      );

      if (assistantMessage) {
        const assistantTimestamp = now();
        deps.actions.addMessage({
          ...deps.logic.createAssistantChatMessage({
            id: createId(assistantTimestamp),
            content: assistantMessage.message,
            createdAt: assistantTimestamp,
          }),
        });
        deps.actions.onMessageReceived(assistantMessage);
      }
    } finally {
      if (!isMountedRef.current) return;
      deps.actions.setSending(false);
      uiState.clearAttachments();
    }
  }, [deps, uiState, now, createId]);

  /**
   * Navigate through input history and update UI value.
   *
   * @param direction - Required history direction (older/newer).
   * @returns void
   */
  const handleHistory = useCallback(
    (direction: ChatHistoryDirection): void => {
      if (deps.state.inputHistory.length === 0) return;
      if (direction === "up" && deps.state.historyCursor === null) {
        draftValueRef.current = uiState.value;
      }

      const nextValue = deps.actions.moveHistoryCursor(direction);
      if (
        deps.logic.shouldRestoreDraftValue({
          direction,
          historyCursor: deps.state.historyCursor,
          nextValue,
        })
      ) {
        uiState.setValue(draftValueRef.current);
        return;
      }

      uiState.setValue(nextValue);
    },
    [deps, uiState]
  );

  /**
   * Reset the history cursor to the "not navigating" state.
   *
   * @returns void
   */
  const resetHistoryCursor = useCallback((): void => {
    deps.actions.resetHistoryCursor();
  }, [deps]);

  /**
   * Update selected model in store state.
   *
   * @param value - Required model identifier (or null to clear selection).
   * @returns void
   */
  const setSelectedModelId = useCallback(
    (value: string | null): void => {
      deps.actions.setSelectedModelId(value);
    },
    [deps]
  );

  /**
   * Fetch model options and apply deterministic selection/fallback behavior.
   * Exposed to support manual refetch and test harnessing.
   *
   * @returns Promise that resolves when fetch and state reconciliation finish.
   */
  const refetchModels = useCallback(async (): Promise<void> => {
    try {
      deps.actions.setModelsLoading(true);
      const result = await deps.api.fetchModels({});

      if (!result || result.status === "error") {
        setFallbackModels({
          actions: deps.actions,
          selectedModelId: deps.state.selectedModelId,
          resolveFallbackModelSelection: deps.logic.resolveFallbackModelSelection,
        });
        return;
      }

      setFetchedModels({
        actions: deps.actions,
        selectedModelId: deps.state.selectedModelId,
        result: {
          currentModel: result.currentModel,
          models: result.models,
        },
        resolveFetchedModelSelection: deps.logic.resolveFetchedModelSelection,
      });
    } catch {
      setFallbackModels({
        actions: deps.actions,
        selectedModelId: deps.state.selectedModelId,
        resolveFallbackModelSelection: deps.logic.resolveFallbackModelSelection,
      });
    } finally {
      deps.actions.setModelsLoading(false);
    }
  }, [deps]);

  return useMemo(
    () => ({
      submit,
      handleHistory,
      resetHistoryCursor,
      setSelectedModelId,
      refetchModels,
    }),
    [submit, handleHistory, resetHistoryCursor, setSelectedModelId, refetchModels]
  );
}

/**
 * Top-level integration hook that composes UI state and orchestration logic.
 *
 * @param deps - Required dependency bag from orchestrator wiring.
 * @returns Combined state/actions API consumed by chat views.
 */
export function useChatIntegration(deps: ChatDeps): ChatIntegration {
  const uiState = useChatUiState();
  const actions = useChatLogic(uiState, deps);
  const { state } = deps;
  const initialModelBootstrapRequestedRef = useRef(false);
  const refetchModelsRef = useRef(actions.refetchModels);

  useEffect(() => {
    if (DEBUG_EFFECTS) {
      console.log("[chat-debug] refetch_models_ref_updated", {
        path: getDebugPath(),
      });
    }
    refetchModelsRef.current = actions.refetchModels;
  }, [actions.refetchModels]);

  /**
   * Bootstraps model options once, only when model options are empty and not loading.
   * Prevents duplicate fetches while preserving lazy initialization.
   */
  useEffect(() => {
    if (DEBUG_EFFECTS) {
      console.log("[chat-debug] bootstrap_models_effect", {
        path: getDebugPath(),
        modelOptionsLength: state.modelOptions.length,
        isModelsLoading: state.isModelsLoading,
        bootstrapRequested: initialModelBootstrapRequestedRef.current,
      });
    }
    if (state.modelOptions.length > 0) {
      initialModelBootstrapRequestedRef.current = false;
      return;
    }
    if (state.isModelsLoading) return;
    if (initialModelBootstrapRequestedRef.current) return;

    initialModelBootstrapRequestedRef.current = true;
    refetchModelsRef.current();
  }, [state.isModelsLoading, state.modelOptions.length]);

  return useMemo(
    () => ({
      ...deps.state,
      ...uiState,
      ...actions,
    }),
    [deps.state, uiState, actions]
  );
}
