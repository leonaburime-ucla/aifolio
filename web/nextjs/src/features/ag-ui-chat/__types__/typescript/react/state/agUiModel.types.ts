export type AgUiModelOption = {
  id: string;
  label: string;
};

export type AgUiModelStoreState = {
  modelOptions: AgUiModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  setModelOptions: (value: AgUiModelOption[]) => void;
  setSelectedModelId: (value: string | null) => void;
  setModelsLoading: (value: boolean) => void;
};

export type AgUiModelStatePort = {
  modelOptions: AgUiModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  setSelectedModelId: (value: string | null) => void;
};
