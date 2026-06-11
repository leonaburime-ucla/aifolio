import { useShallow } from "zustand/react/shallow";
import type { AgUiModelStatePort } from "@aifolio/contracts/entities/ag-ui";
import { useAgUiModelStore } from "@/features/ag-ui-chat/react/state/zustand/agUiModelStore";

export function useAgUiModelStateAdapter(): AgUiModelStatePort {
  const { modelOptions, selectedModelId, isModelsLoading, backendError, setSelectedModelId } =
    useAgUiModelStore(
      useShallow((state) => ({
        modelOptions: state.modelOptions,
        selectedModelId: state.selectedModelId,
        isModelsLoading: state.isModelsLoading,
        backendError: state.backendError,
        setSelectedModelId: state.setSelectedModelId,
      }))
    );

  return {
    modelOptions,
    selectedModelId,
    isModelsLoading,
    backendError,
    setSelectedModelId,
  };
}
