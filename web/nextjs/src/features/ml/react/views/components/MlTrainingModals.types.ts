import type { HyperParams } from "@aifolio/contracts/entities/ml-training";
import type {
  DistillComparison,
  TrainingMetrics,
} from "@aifolio/contracts/entities/ml-training";
import type { OptimalPrediction } from "@aifolio/contracts/entities/ml-training";

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
