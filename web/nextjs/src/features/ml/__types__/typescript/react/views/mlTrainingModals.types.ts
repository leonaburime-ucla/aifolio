import type { HyperParams } from "@/features/ml/__types__/typescript/utils/bayesianOptimizer.types";
import type {
  DistillComparison,
  TrainingMetrics,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
import type { OptimalPrediction } from "@/features/ml/__types__/typescript/react/hooks/mlTrainingUiBase.types";

export type OptimalParamsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  pendingOptimalParams: HyperParams | null;
  pendingOptimalPrediction: OptimalPrediction | null;
  onApply: () => void;
  activeAlgorithm?: string;
};

export type DistillMetricsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  distillMetrics: TrainingMetrics | null;
  distillModelId: string | null;
  distillModelPath: string | null;
  distillComparison?: DistillComparison | null;
};
