import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";
import type { AgUiWorkspaceTab } from "@/features/ag-ui-chat/__types__/typescript/react/state/agUiWorkspace.types";

/**
 * AG-UI chat-local frontend tool contracts (route/chart concerns).
 *
 * ML training tool contracts are owned by `features/ml` and re-exported below.
 */

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

export type {
  EnsureFrameworkTabArgs,
  MlFormBridge,
  MlFormPatch,
  MlFormPatchResult,
  MlFormRandomizeArgs,
  MlFormStartResult,
  MlFrameworkTab,
  MlListField,
  MlMode,
  MlTask,
  PytorchFormBridge,
  PytorchFormPatch,
  PytorchRandomizeArgs,
  PytorchTrainingMode,
  TensorflowFormBridge,
  TensorflowFormPatch,
  TensorflowRandomizeArgs,
  TensorflowTrainingMode,
  TrainPytorchModelArgs,
  TrainTensorflowModelArgs,
} from "@/features/ml/__types__/typescript/ai/agUi/mlTrainingTooling.types";
