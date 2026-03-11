export type AgUiModelOption = {
  id: string;
  label: string;
};

export type AgUiModelStoreState = {
  modelOptions: AgUiModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  backendError: string | null;
  setModelOptions: (value: AgUiModelOption[]) => void;
  setSelectedModelId: (value: string | null) => void;
  setModelsLoading: (value: boolean) => void;
  setBackendError: (value: string | null) => void;
};

export type AgUiModelStatePort = {
  modelOptions: AgUiModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  backendError: string | null;
  setSelectedModelId: (value: string | null) => void;
};
