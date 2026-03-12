import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  ChatActions,
  ChatAttachment,
  ChatDeps,
  ChatHistoryDirection,
  ChatHistoryMessage,
  ChatIntegration,
  ChatModelOption,
  ChatStateActions,
  ChatUiState,
} from "@/features/ai-chat/__types__/typescript/chat.types";
import type { ScreenFeedback } from "@/features/ai-chat/__types__/typescript/uiFeedback.types";

const DEBUG_EFFECTS = process.env.NEXT_PUBLIC_DEBUG_EFFECTS === "1";
const INVALID_RESPONSE_FEEDBACK: ScreenFeedback = {
  kind: "error",
  code: "CHAT_RESPONSE_INVALID",
  message: "The AI service did not return a usable response. Try again.",
  retryable: true,
  actionLabel: "Try again",
};

function getDebugPath(): string {
  return globalThis.location?.pathname ?? "";
}

type PendingSubmission = {
  value: string;
  model: string | null;
  history: ChatHistoryMessage[];
  attachments: ChatAttachment[];
  datasetId: string | null;
};

/**
 * Convert unknown request failures into a stable, UI-safe feedback contract.
 *
 * @param error - Required unknown runtime failure.
 * @returns Screen feedback that the view can render persistently.
 */
function resolveSubmitFeedback(error: unknown): ScreenFeedback {
  if (
    error &&
    typeof error === "object" &&
    "code" in error &&
    error.code === "CHAT_REQUEST_HTTP_ERROR" &&
    "status" in error &&
    typeof error.status === "number"
  ) {
    return {
      kind: "error",
      code: error.status >= 500 ? "CHAT_SERVICE_UNAVAILABLE" : "CHAT_REQUEST_REJECTED",
      message:
        error.status >= 500
          ? "The AI service returned an error. Try again in a moment."
          : "The AI service rejected the request. Check the request and try again.",
      retryable: true,
      actionLabel: "Try again",
    };
  }

  if (
    error &&
    typeof error === "object" &&
    "code" in error &&
    error.code === "CHAT_RESPONSE_PARSE_ERROR"
  ) {
    return {
      kind: "error",
      code: "CHAT_RESPONSE_INVALID",
      message: "The AI service returned an unreadable response. Try again.",
      retryable: true,
      actionLabel: "Try again",
    };
  }

  if (error instanceof DOMException && error.name === "AbortError") {
    return {
      kind: "info",
      code: "CHAT_REQUEST_ABORTED",
      message: "The request was canceled before a response was returned.",
    };
  }

  if (globalThis.navigator?.onLine === false) {
    return {
      kind: "error",
      code: "CHAT_OFFLINE",
      message: "You're offline. Reconnect to the internet and try again.",
      retryable: true,
      actionLabel: "Try again",
    };
  }

  return {
    kind: "error",
    code: "CHAT_REQUEST_FAILED",
    message: "Could not reach the AI service. Check your connection and try again.",
    retryable: true,
    actionLabel: "Try again",
  };
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
  const lastSubmissionRef = useRef<PendingSubmission | null>(null);
  const { state, actions, api, logic } = deps;

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
  const runSubmission = useCallback(
    async (submission: PendingSubmission): Promise<void> => {
      actions.setScreenFeedback(null);
      actions.setSending(true);
      try {
        const assistantMessage = await api.sendMessage(
          {
            value: submission.value,
            model: submission.model,
            history: submission.history,
            attachments: submission.attachments,
          },
          {
            datasetId: submission.datasetId,
          }
        );

        if (!isMountedRef.current) return;

        if (!assistantMessage) {
          actions.setScreenFeedback(INVALID_RESPONSE_FEEDBACK);
          return;
        }

        actions.setScreenFeedback(null);

        const assistantTimestamp = now();
        actions.addMessage({
          ...logic.createAssistantChatMessage({
            id: createId(assistantTimestamp),
            content: assistantMessage.message,
            createdAt: assistantTimestamp,
          }),
        });
        actions.onMessageReceived(assistantMessage);
      } catch (error) {
        if (!isMountedRef.current) return;
        actions.setScreenFeedback(resolveSubmitFeedback(error));
      } finally {
        if (!isMountedRef.current) return;
        actions.setSending(false);
      }
    },
    [actions, api, logic, now, createId]
  );

  const submit = useCallback(async (): Promise<void> => {
    const trimmed = logic.normalizeSubmissionValue({ value: uiState.value });
    if (!trimmed) return;

    actions.addInputToHistory(trimmed);
    const history = logic.buildChatHistoryWindow({
      messages: state.messages,
      userContent: trimmed,
      attachments: uiState.attachments,
    });

    const userTimestamp = now();
    actions.addMessage({
      ...logic.createUserChatMessage({
        id: createId(userTimestamp),
        content: trimmed,
        createdAt: userTimestamp,
      }),
    });

    lastSubmissionRef.current = {
      value: trimmed,
      model: state.selectedModelId,
      history,
      attachments: [...uiState.attachments],
      datasetId: state.activeDatasetId ?? null,
    };

    uiState.resetValue();
    actions.resetHistoryCursor();
    uiState.clearAttachments();

    await runSubmission(lastSubmissionRef.current);
  }, [actions, logic, state.messages, state.selectedModelId, state.activeDatasetId, uiState, now, createId, runSubmission]);

  const retryLastSubmission = useCallback(async (): Promise<void> => {
    if (!lastSubmissionRef.current) return;
    await runSubmission(lastSubmissionRef.current);
  }, [runSubmission]);

  /**
   * Navigate through input history and update UI value.
   *
   * @param direction - Required history direction (older/newer).
   * @returns void
   */
  const handleHistory = useCallback(
    (direction: ChatHistoryDirection): void => {
      if (deps.state.inputHistory.length === 0) return;
      if (direction === "up" && state.historyCursor === null) {
        draftValueRef.current = uiState.value;
      }

      const nextValue = actions.moveHistoryCursor(direction);
      if (
        logic.shouldRestoreDraftValue({
          direction,
          historyCursor: state.historyCursor,
          nextValue,
        })
      ) {
        uiState.setValue(draftValueRef.current);
        return;
      }

      uiState.setValue(nextValue);
    },
    [actions, logic, state.inputHistory.length, state.historyCursor, uiState]
  );

  /**
   * Reset the history cursor to the "not navigating" state.
   *
   * @returns void
   */
  const resetHistoryCursor = useCallback((): void => {
    actions.resetHistoryCursor();
  }, [actions]);

  /**
   * Update selected model in store state.
   *
   * @param value - Required model identifier (or null to clear selection).
   * @returns void
   */
  const setSelectedModelId = useCallback(
    (value: string | null): void => {
      actions.setSelectedModelId(value);
    },
    [actions]
  );

  /**
   * Update persistent inline feedback for the current chat surface.
   *
   * @param value - Feedback object to persist, or null to clear it.
   * @returns void
   */
  const setScreenFeedback = useCallback(
    (value: ScreenFeedback | null): void => {
      actions.setScreenFeedback(value);
    },
    [actions]
  );

  /**
   * Fetch model options and apply deterministic selection/fallback behavior.
   * Exposed to support manual refetch and test harnessing.
   *
   * @returns Promise that resolves when fetch and state reconciliation finish.
   */
  const refetchModels = useCallback(async (): Promise<void> => {
    try {
      actions.setModelsLoading(true);
      const result = await api.fetchModels({});

      if (!result || result.status === "error") {
        setFallbackModels({
          actions,
          selectedModelId: state.selectedModelId,
          resolveFallbackModelSelection: logic.resolveFallbackModelSelection,
        });
        return;
      }

      if (!isMountedRef.current) return;

      setFetchedModels({
        actions,
        selectedModelId: state.selectedModelId,
        result: {
          currentModel: result.currentModel,
          models: result.models,
        },
        resolveFetchedModelSelection: logic.resolveFetchedModelSelection,
      });
    } catch {
      setFallbackModels({
        actions,
        selectedModelId: state.selectedModelId,
        resolveFallbackModelSelection: logic.resolveFallbackModelSelection,
      });
    } finally {
      if (!isMountedRef.current) return;
      actions.setModelsLoading(false);
    }
  }, [actions, api, state.selectedModelId, logic]);

  return useMemo(
    () => ({
      submit,
      retryLastSubmission,
      handleHistory,
      resetHistoryCursor,
      setSelectedModelId,
      setScreenFeedback,
      refetchModels,
    }),
    [
      submit,
      retryLastSubmission,
      handleHistory,
      resetHistoryCursor,
      setSelectedModelId,
      setScreenFeedback,
      refetchModels,
    ]
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
