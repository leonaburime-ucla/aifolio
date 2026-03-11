"use client";

import { useMemo } from "react";
import { fetchAgUiModels } from "@/features/ag-ui-chat/typescript/api/agUiModelApi";
import { resolveNextAgUiSelectedModelId } from "@/features/ag-ui-chat/typescript/logic/agUiContext.logic";
import { useAgUiModelStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiModelState.adapter";
import { useAgUiModelStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/agUiModelStore";
import {
  useAgUiModelSelector,
  type AgUiModelSelectorDeps,
} from "@/features/ag-ui-chat/typescript/react/hooks/useAgUiModelSelector.hooks";

/**
 * Orchestrator hook for the AG-UI model selector.
 *
 * Wires injected dependencies (state adapter, store actions, API, logic)
 * into the pure `useAgUiModelSelector` hook so the hook itself never
 * imports concrete stores or fetch functions.
 *
 * @returns View model consumed by `AgUiModelSelector` component.
 */
export function useAgUiModelSelectorOrchestrator() {
  const statePort = useAgUiModelStateAdapter();

  const state = useMemo(
    () => ({
      modelOptions: statePort.modelOptions,
      selectedModelId: statePort.selectedModelId,
      isModelsLoading: statePort.isModelsLoading,
      backendError: statePort.backendError,
    }),
    [statePort.modelOptions, statePort.selectedModelId, statePort.isModelsLoading, statePort.backendError]
  );

  const actions = useMemo(
    () => ({
      setModelOptions: useAgUiModelStore.getState().setModelOptions,
      setSelectedModelId: useAgUiModelStore.getState().setSelectedModelId,
      setModelsLoading: useAgUiModelStore.getState().setModelsLoading,
      setBackendError: useAgUiModelStore.getState().setBackendError,
    }),
    []
  );

  const api = useMemo(
    () => ({
      fetchModels: fetchAgUiModels,
      resolveSelectedModelId: resolveNextAgUiSelectedModelId,
    }),
    []
  );

  const deps = useMemo<AgUiModelSelectorDeps>(
    () => ({ state, actions, api }),
    [state, actions, api]
  );

  return useAgUiModelSelector(deps);
}
