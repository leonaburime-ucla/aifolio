import { create } from "zustand";
import type {
  AgUiModelStoreState,
} from "@aifolio/contracts/entities/ag-ui";
import { AG_UI_FALLBACK_MODELS } from "@aifolio/frontend-core/ag-ui";

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
