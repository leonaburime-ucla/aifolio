import type { ChartSpec } from "../../chart/model/types";
import type { MlFrameworkTab } from "../../ml-training/model/types";

export type CopilotAgentName = "agentic-research";

export type CopilotRuntimeConfig = {
  runtimeUrl: string;
  agent: CopilotAgentName;
  backendBaseUrl: string;
  backendAguiPath: string;
};

export type AgUiWorkspaceTab = "charts" | "agentic-research" | "pytorch" | "tensorflow";

export type AgUiWorkspaceStoreState = {
  activeTab: AgUiWorkspaceTab;
  setActiveTab: (tab: AgUiWorkspaceTab) => void;
};

export type AgUiWorkspaceStatePort = {
  activeTab: AgUiWorkspaceTab;
  setActiveTab: (tab: AgUiWorkspaceTab) => void;
};

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

export type CopilotAssistantPayload = {
  type?: "TextMessage";
  message: string;
  chartSpec: ChartSpec | ChartSpec[] | null;
};

export type AddChartSpecPayload = {
  chartSpec?: unknown;
  chartSpecs?: unknown[];
};

export type NavigateToPageResult =
  | { status: "ok"; resolvedRoute: string }
  | { status: "error"; code: "INVALID_ROUTE"; allowedRoutes: string[] };

export type SwitchAgUiTabResult =
  | { status: "ok"; tab: AgUiWorkspaceTab }
  | { status: "error"; code: "INVALID_TAB"; allowedTabs: AgUiWorkspaceTab[] };

export type AddChartSpecHandler = (spec: ChartSpec) => void;

export type CopilotActionParameterType =
  | "string"
  | "number"
  | "boolean"
  | "object"
  | "string[]"
  | "number[]"
  | "boolean[]"
  | "object[]";

export type CopilotActionParameter = {
  name: string;
  type: CopilotActionParameterType;
  required: boolean;
  description: string;
};

export type AgUiTabSwitchArgs = {
  tab: string;
};

export type NavigateToPageArgs = {
  route: string;
};

export type EnsureFrameworkTabArgs = {
  activeTab: string;
  setActiveTab: (tab: MlFrameworkTab) => void;
  pushRoute: (route: string) => void;
  frameworkTab: MlFrameworkTab;
  waitForFrameworkForm: () => Promise<boolean>;
};

export type EnsurePytorchTabArgs = Omit<EnsureFrameworkTabArgs, "frameworkTab" | "waitForFrameworkForm"> & {
  waitForPytorchForm: () => Promise<boolean>;
};

export type EnsureTensorflowTabArgs = Omit<EnsureFrameworkTabArgs, "frameworkTab" | "waitForFrameworkForm"> & {
  waitForTensorflowForm: () => Promise<boolean>;
};
