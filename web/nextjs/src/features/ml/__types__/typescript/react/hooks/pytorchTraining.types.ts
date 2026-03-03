import type { useMlDatasetOrchestrator } from "@/features/ml/typescript/react/orchestrators/mlDatasetOrchestrator";
import type { MlTrainingUiBaseState } from "@/features/ml/__types__/typescript/react/hooks/mlTrainingUiBase.types";
import type { PytorchTrainingMode } from "@/features/ml/__types__/typescript/api/pytorchApi.types";
import type {
  DistillComparison,
  TrainingMetrics,
  TrainingRunRow,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type {
  runPytorchDistillation,
  runPytorchTraining,
} from "@/features/ml/typescript/react/orchestrators/pytorchTraining.orchestrator";
import type { useMlTrainingRunsAdapter } from "@/features/ml/typescript/react/state/adapters/mlTrainingRuns.adapter";

export type PytorchUiState = MlTrainingUiBaseState & {
  trainingMode: PytorchTrainingMode;
  setTrainingMode: (value: PytorchTrainingMode) => void;
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

export type PytorchLogicArgs = {
  dataset: ReturnType<typeof useMlDatasetOrchestrator>;
  trainingRuns: TrainingRunRow[];
  prependTrainingRun: (row: TrainingRunRow) => void;
  ui: PytorchUiState;
  runTraining: typeof runPytorchTraining;
  runDistillation: typeof runPytorchDistillation;
  runtime?: Partial<PytorchRuntimeDeps>;
};

export type PytorchRuntimeDeps = {
  notifySuccess: (message: string) => void;
  notifyError: (message: string) => void;
  schedule: (callback: () => void, delayMs: number) => void;
  writeClipboardText: (text: string) => Promise<void>;
};

export type PytorchIntegrationArgs = {
  useTrainingRunsState?: typeof useMlTrainingRunsAdapter;
  runTraining?: typeof runPytorchTraining;
  runDistillation?: typeof runPytorchDistillation;
  runtime?: Partial<PytorchRuntimeDeps>;
};
