import type { MlTaskType } from "@/features/ml/__types__/typescript/config/datasetTrainingDefaults.types";
import type { PytorchTrainingMode } from "@/features/ml/__types__/typescript/api/pytorchApi.types";

/**
 * Patch payload accepted by the PyTorch form bridge.
 * Keys intentionally mirror AI tool-call field names.
 */
export type PytorchBridgePatch = {
  dataset_id?: string;
  training_mode?: PytorchTrainingMode;
  target_column?: string;
  task?: MlTaskType;
  epoch_values?: string | number | number[];
  batch_sizes?: string | number | number[];
  learning_rates?: string | number | number[];
  test_sizes?: string | number | number[];
  hidden_dims?: string | number | number[];
  num_hidden_layers?: string | number | number[];
  dropouts?: string | number | number[];
  exclude_columns?: string | string[];
  date_columns?: string | string[];
  set_sweep_values?: boolean;
  run_sweep?: boolean;
  auto_distill?: boolean;
};

export type PytorchBridgeApplyResult = {
  applied: string[];
  skipped: string[];
};

export type PytorchBridgeStartResult =
  | { status: "ok" }
  | { status: "error"; reason: string };

/**
 * Global bridge surface exposed for external AI-driven form automation.
 */
export type PytorchFormBridge = {
  applyPatch: (patch: PytorchBridgePatch) => PytorchBridgeApplyResult;
  startTrainingRuns: () => Promise<PytorchBridgeStartResult>;
};

/**
 * Minimal patch-application dependencies for pure patch processing.
 */
export type PytorchBridgePatchBindings = {
  setDatasetId: (value: string) => void;
  setTrainingMode: (value: PytorchTrainingMode) => void;
  setTargetColumn: (value: string) => void;
  setTask: (value: MlTaskType) => void;
  runSweepEnabled: boolean;
  toggleRunSweep: (enabled: boolean) => void;
  setEpochValuesInput: (value: string) => void;
  setBatchSizesInput: (value: string) => void;
  setLearningRatesInput: (value: string) => void;
  setTestSizesInput: (value: string) => void;
  setHiddenDimsInput: (value: string) => void;
  setNumHiddenLayersInput: (value: string) => void;
  setDropoutsInput: (value: string) => void;
  setExcludeColumnsInput: (value: string) => void;
  setDateColumnsInput: (value: string) => void;
  autoDistillEnabled: boolean;
  setAutoDistillEnabled: (enabled: boolean) => void;
};

/**
 * Setter/action bindings needed by the bridge hook.
 * This keeps all bridge logic outside route files so pages remain mostly JSX.
 */
export type PytorchFormBridgeBindings = PytorchBridgePatchBindings & {
  trainingMode: PytorchTrainingMode;
  onTrainClick: () => Promise<void>;
};
