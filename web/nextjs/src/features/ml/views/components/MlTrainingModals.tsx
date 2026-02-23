import { Modal } from "@/core/views/components/General/Modal";
import type { HyperParams } from "@/features/ml/utils/bayesianOptimizer.util";
import {
  formatMetricNumber,
  type TrainingMetrics,
} from "@/features/ml/utils/trainingRuns.util";

type OptimalParamsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  pendingOptimalParams: HyperParams | null;
  pendingOptimalPrediction: { metricName: string; metricValue: number } | null;
  onApply: () => void;
};

export function OptimalParamsModal({
  isOpen,
  onClose,
  pendingOptimalParams,
  pendingOptimalPrediction,
  onApply,
}: OptimalParamsModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Bayesian Optimization Suggestion">
      <div className="space-y-4 p-1">
        <p className="text-sm text-zinc-600">
          Suggested next hyperparameters for better accuracy based on completed runs.
        </p>
        <div className="grid grid-cols-1 gap-2 text-sm text-zinc-800 md:grid-cols-2">
          <p>epochs: <span className="font-semibold">{pendingOptimalParams?.epochs ?? "n/a"}</span></p>
          <p>learning_rate: <span className="font-semibold">{pendingOptimalParams ? Number(pendingOptimalParams.learning_rate.toPrecision(6)) : "n/a"}</span></p>
          <p>test_size: <span className="font-semibold">{pendingOptimalParams ? Number(pendingOptimalParams.test_size.toPrecision(4)) : "n/a"}</span></p>
          <p>batch_size: <span className="font-semibold">{pendingOptimalParams?.batch_size ?? "n/a"}</span></p>
          <p>hidden_dim: <span className="font-semibold">{pendingOptimalParams?.hidden_dim ?? "n/a"}</span></p>
          <p>num_hidden_layers: <span className="font-semibold">{pendingOptimalParams?.num_hidden_layers ?? "n/a"}</span></p>
          <p>dropout: <span className="font-semibold">{pendingOptimalParams ? Number(pendingOptimalParams.dropout.toPrecision(4)) : "n/a"}</span></p>
        </div>
        {pendingOptimalPrediction ? (
          <p className="text-sm font-semibold text-red-600">
            Predicted: {pendingOptimalPrediction.metricName} ≈{" "}
            {formatMetricNumber(pendingOptimalPrediction.metricValue)}
          </p>
        ) : null}
        <div className="flex items-center justify-end gap-2 pt-2">
          <button
            type="button"
            className="rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            type="button"
            className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white"
            onClick={onApply}
            disabled={!pendingOptimalParams}
          >
            Update Table With Values
          </button>
        </div>
      </div>
    </Modal>
  );
}

type DistillMetricsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  distillMetrics: TrainingMetrics | null;
  distillModelId: string | null;
  distillModelPath: string | null;
};

export function DistillMetricsModal({
  isOpen,
  onClose,
  distillMetrics,
  distillModelId,
  distillModelPath,
}: DistillMetricsModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Distillation Metrics">
      <div className="space-y-3 p-1 text-sm text-zinc-700">
        <p>
          metric_name:{" "}
          <span className="font-semibold text-zinc-900">
            {distillMetrics?.test_metric_name ?? "n/a"}
          </span>
        </p>
        <p>
          metric_score:{" "}
          <span className="font-semibold text-zinc-900">
            {formatMetricNumber(distillMetrics?.test_metric_value)}
          </span>
        </p>
        <p>
          train_loss:{" "}
          <span className="font-semibold text-zinc-900">
            {formatMetricNumber(distillMetrics?.train_loss)}
          </span>
        </p>
        <p>
          test_loss:{" "}
          <span className="font-semibold text-zinc-900">
            {formatMetricNumber(distillMetrics?.test_loss)}
          </span>
        </p>
        {distillModelId || distillModelPath ? (
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-600">
            <p>
              model_id: <span className="font-medium text-zinc-800">{distillModelId ?? "n/a"}</span>
            </p>
            <p className="mt-1 break-all">
              model_path: <span className="font-medium text-zinc-800">{distillModelPath ?? "n/a"}</span>
            </p>
          </div>
        ) : (
          <p className="text-xs text-zinc-500">
            Model files were not saved for this run.
          </p>
        )}
        <div className="flex justify-end pt-1">
          <button
            type="button"
            className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
    </Modal>
  );
}
