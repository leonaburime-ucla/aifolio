import type { MlTaskType } from "@/features/ml/__types__/typescript/config/datasetTrainingDefaults.types";
import type { TensorflowTrainingMode } from "@/features/ml/__types__/typescript/api/tensorflowApi.types";

/**
 * Patch payload accepted by the TensorFlow form bridge.
 * Keys intentionally mirror AI tool-call field names.
 */
export type TensorflowBridgePatch = {
  dataset_id?: string;
  training_mode?: TensorflowTrainingMode;
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

export type TensorflowBridgeApplyResult = {
  applied: string[];
  skipped: string[];
};

export type TensorflowBridgeStartResult =
  | { status: "ok" }
  | { status: "error"; reason: string };

export type TensorflowFormBridge = {
  applyPatch: (patch: TensorflowBridgePatch) => TensorflowBridgeApplyResult;
  startTrainingRuns: () => Promise<TensorflowBridgeStartResult>;
};

export type TensorflowBridgePatchBindings = {
  setDatasetId: (value: string) => void;
  setTrainingMode: (value: TensorflowTrainingMode) => void;
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

export type TensorflowFormBridgeBindings = TensorflowBridgePatchBindings & {
  trainingMode: TensorflowTrainingMode;
  onTrainClick: () => Promise<void>;
};
