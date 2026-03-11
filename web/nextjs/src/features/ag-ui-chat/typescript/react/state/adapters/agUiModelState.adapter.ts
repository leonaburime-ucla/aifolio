import { useShallow } from "zustand/react/shallow";
import type { AgUiModelStatePort } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiModel.types";
import { useAgUiModelStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/agUiModelStore";

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
