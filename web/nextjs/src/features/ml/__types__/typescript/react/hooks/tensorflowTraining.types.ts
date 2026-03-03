import type { useMlDatasetOrchestrator } from "@/features/ml/typescript/react/orchestrators/mlDatasetOrchestrator";
import type { MlTrainingUiBaseState } from "@/features/ml/__types__/typescript/react/hooks/mlTrainingUiBase.types";
import type { TensorflowTrainingMode } from "@/features/ml/__types__/typescript/api/tensorflowApi.types";
import type {
  DistillComparison,
  TrainingMetrics,
  TrainingRunRow,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type {
  runTensorflowDistillation,
  runTensorflowTraining,
} from "@/features/ml/typescript/react/orchestrators/tensorflowTraining.orchestrator";
import type { useMlTrainingRunsAdapter } from "@/features/ml/typescript/react/state/adapters/mlTrainingRuns.adapter";

export type TensorflowUiState = MlTrainingUiBaseState & {
  trainingMode: TensorflowTrainingMode;
  setTrainingMode: (value: TensorflowTrainingMode) => void;
};

export type DistilledSnapshotByTeacher = Record<
  string,
  {
    metrics: TrainingMetrics;
    modelId: string | null;
    modelPath: string | null;
    comparison: DistillComparison;
  }
>;

export type TensorflowLogicArgs = {
  dataset: ReturnType<typeof useMlDatasetOrchestrator>;
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  ui: TensorflowUiState;
  runTraining: typeof runTensorflowTraining;
  runDistillation: typeof runTensorflowDistillation;
  runtime?: Partial<TensorflowRuntimeDeps>;
};

export type TensorflowRuntimeDeps = {
  notifySuccess: (message: string) => void;
  notifyError: (message: string) => void;
  schedule: (callback: () => void, delayMs: number) => void;
  writeClipboardText: (text: string) => Promise<void>;
};

export type TensorflowIntegrationArgs = {
  useTrainingRunsState?: typeof useMlTrainingRunsAdapter;
  runTraining?: typeof runTensorflowTraining;
  runDistillation?: typeof runTensorflowDistillation;
  runtime?: Partial<TensorflowRuntimeDeps>;
};
