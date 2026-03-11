"use client";

import { useEffect, useRef } from "react";
import type { AgUiModelOption } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiModel.types";

// ---------------------------------------------------------------------------
// Dependency Injection types
// ---------------------------------------------------------------------------

export type AgUiModelSelectorState = {
  modelOptions: AgUiModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  backendError: string | null;
};

export type AgUiModelSelectorActions = {
  setModelOptions: (value: AgUiModelOption[]) => void;
  setSelectedModelId: (value: string | null) => void;
  setModelsLoading: (value: boolean) => void;
  setBackendError: (value: string | null) => void;
};

export type AgUiModelSelectorApi = {
  fetchModels: () => Promise<{ currentModel: string | null; models: AgUiModelOption[] } | null>;
  resolveSelectedModelId: (params: {
    currentSelectedModelId: string | null;
    fetchedModels: AgUiModelOption[];
    apiCurrentModelId: string | null;
  }) => string | null;
};

export type AgUiModelSelectorDeps = {
  state: AgUiModelSelectorState;
  actions: AgUiModelSelectorActions;
  api: AgUiModelSelectorApi;
};

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Encapsulates AG-UI model selector loading and selection behavior.
 *
 * All dependencies (state reads, state writes, and API calls) are injected
 * via the `deps` bag so the hook is fully testable and respects DI.
 *
 * @param deps - Required injected dependency bag from orchestrator wiring.
 * @returns Selector view model and actions for the model dropdown component.
 */
export function useAgUiModelSelector(deps: AgUiModelSelectorDeps) {
  const { state, actions, api } = deps;
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function loadModels() {
      hasLoadedRef.current = true;
      actions.setModelsLoading(true);
      try {
        const result = await api.fetchModels();
        if (cancelled) return;

        if (!result) {
          actions.setBackendError(
            "Cannot connect to backend — make sure the backend server is running."
          );
          return;
        }

        actions.setBackendError(null);
        actions.setModelOptions(result.models);

        const nextSelected = api.resolveSelectedModelId({
          currentSelectedModelId: state.selectedModelId,
          fetchedModels: result.models,
          apiCurrentModelId: result.currentModel ?? null,
        });
        actions.setSelectedModelId(nextSelected);
      } finally {
        if (!cancelled) {
          actions.setModelsLoading(false);
        }
      }
    }

    if (!state.isModelsLoading && !hasLoadedRef.current) {
      void loadModels();
    }

    return () => {
      cancelled = true;
    };
  }, [state.isModelsLoading, actions, api]);

  return {
    modelOptions: state.modelOptions,
    selectedModelId: state.selectedModelId,
    isModelsLoading: state.isModelsLoading,
    backendError: state.backendError,
    setSelectedModelId: actions.setSelectedModelId,
  };
}
