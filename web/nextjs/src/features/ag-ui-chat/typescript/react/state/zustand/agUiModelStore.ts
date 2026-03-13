import { create } from "zustand";
import type {
  AgUiModelStoreState,
} from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiModel.types";
import { AG_UI_FALLBACK_MODELS } from "@/features/ag-ui-chat/typescript/config/agUiModel.config";

export const useAgUiModelStore = create<AgUiModelStoreState>((set) => ({
  modelOptions: AG_UI_FALLBACK_MODELS,
  selectedModelId: AG_UI_FALLBACK_MODELS[0]?.id ?? null,
  isModelsLoading: false,
  backendError: null,
  setModelOptions: (value) => set(() => ({ modelOptions: value })),
  setSelectedModelId: (value) => set(() => ({ selectedModelId: value })),
  setModelsLoading: (value) => set(() => ({ isModelsLoading: value })),
  setBackendError: (value) => set(() => ({ backendError: value })),
}));

export type { AgUiModelStoreState };
