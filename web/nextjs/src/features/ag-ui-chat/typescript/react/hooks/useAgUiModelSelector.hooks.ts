"use client";

import { useEffect, useRef } from "react";
import { fetchAgUiModels } from "@/features/ag-ui-chat/typescript/api/agUiModelApi";
import { resolveNextAgUiSelectedModelId } from "@/features/ag-ui-chat/typescript/logic/agUiContext.logic";
import { useAgUiModelStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/agUiModelStore";
import { useAgUiModelStateAdapter } from "@/features/ag-ui-chat/typescript/react/state/adapters/agUiModelState.adapter";

/**
 * Encapsulates AG-UI model selector loading and selection behavior.
 *
 * @returns Selector view model and actions for the model dropdown component.
 */
export function useAgUiModelSelector() {
  const { modelOptions, selectedModelId, isModelsLoading, setSelectedModelId } =
    useAgUiModelStateAdapter();
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function loadModels() {
      hasLoadedRef.current = true;
      useAgUiModelStore.getState().setModelsLoading(true);
      try {
        const result = await fetchAgUiModels();
        if (!result || cancelled) return;

        useAgUiModelStore.getState().setModelOptions(result.models);

        const currentSelected = useAgUiModelStore.getState().selectedModelId;
        const nextSelected = resolveNextAgUiSelectedModelId({
          currentSelectedModelId: currentSelected,
          fetchedModels: result.models,
          apiCurrentModelId: result.currentModel ?? null,
        });
        useAgUiModelStore.getState().setSelectedModelId(nextSelected);
      } finally {
        if (!cancelled) {
          useAgUiModelStore.getState().setModelsLoading(false);
        }
      }
    }

    if (!isModelsLoading && !hasLoadedRef.current) {
      void loadModels();
    }

    return () => {
      cancelled = true;
    };
  }, [isModelsLoading]);

  return {
    modelOptions,
    selectedModelId,
    isModelsLoading,
    setSelectedModelId,
  };
}
