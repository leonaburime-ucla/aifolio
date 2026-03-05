import { Modal } from "@/core/views/components/General/Modal";
import { formatBytes, formatInt } from "@/features/ml/typescript/utils/displayFormat.util";
import { formatMetricNumber } from "@/features/ml/typescript/utils/trainingRuns.util";
import {
  formatPercentLabel,
  hasModelArtifacts,
} from "@/features/ml/typescript/logic/mlTrainingModals.logic";
import type {
  DistillMetricsModalProps,
  OptimalParamsModalProps,
} from "@/features/ml/__types__/typescript/react/views/mlTrainingModals.types";

export function OptimalParamsModal({
  isOpen,
  onClose,
  pendingOptimalParams,
  pendingOptimalPrediction,
  onApply,
  activeAlgorithm,
}: OptimalParamsModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Bayesian Optimization Suggestion">
      <div className="space-y-4 p-1">
        <p className="text-sm text-zinc-600">
          Suggested next hyperparameters for {activeAlgorithm ? <span className="font-semibold text-zinc-800">{activeAlgorithm}</span> : "this architecture"} for better accuracy based on completed runs.
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
            {formatMetricNumber({ value: pendingOptimalPrediction.metricValue })}
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

export function DistillMetricsModal({
  isOpen,
  onClose,
  distillMetrics,
  distillModelId,
  distillModelPath,
  distillComparison,
}: DistillMetricsModalProps) {
  const sizeSavedLabel = formatPercentLabel({
    value: distillComparison?.sizeSavedPercent,
    fallback: "(file-size savings unavailable when no artifact files are persisted)",
  });
  const paramsSavedLabel = formatPercentLabel({
    value: distillComparison?.paramSavedPercent,
    fallback: "",
  });
  const showModelArtifacts = hasModelArtifacts({
    modelId: distillModelId,
    modelPath: distillModelPath,
  });

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Distillation Metrics">
      <div className="space-y-3 p-1 text-sm text-zinc-700">
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-2">
            <p className="text-[11px] uppercase tracking-wide text-zinc-500">metric_name</p>
            <p className="mt-1 font-semibold text-zinc-900">{distillMetrics?.test_metric_name ?? "n/a"}</p>
          </div>
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-2">
            <p className="text-[11px] uppercase tracking-wide text-zinc-500">metric_score</p>
            <p className="mt-1 font-semibold text-zinc-900">
              {formatMetricNumber({ value: distillMetrics?.test_metric_value })}
            </p>
          </div>
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-2">
            <p className="text-[11px] uppercase tracking-wide text-zinc-500">train_loss</p>
            <p className="mt-1 font-semibold text-zinc-900">{formatMetricNumber({ value: distillMetrics?.train_loss })}</p>
          </div>
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-2">
            <p className="text-[11px] uppercase tracking-wide text-zinc-500">test_loss</p>
            <p className="mt-1 font-semibold text-zinc-900">{formatMetricNumber({ value: distillMetrics?.test_loss })}</p>
          </div>
        </div>
        {distillComparison ? (
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700">
            <p className="font-semibold text-zinc-800">Teacher vs Student</p>
            <p className="mt-1">
              metric ({distillComparison.metricName}): teacher{" "}
              <span className="font-medium text-zinc-900">
                {formatMetricNumber({ value: distillComparison.teacherMetricValue })}
              </span>{" "}
              | student{" "}
              <span className="font-medium text-zinc-900">
                {formatMetricNumber({ value: distillComparison.studentMetricValue })}
              </span>
            </p>
            <p className="mt-1">
              quality delta (student vs teacher):{" "}
              <span className="font-medium text-zinc-900">
                {formatMetricNumber({ value: distillComparison.qualityDelta })}
              </span>{" "}
              <span className="text-zinc-500">
                ({distillComparison.higherIsBetter ? "higher is better" : "lower is better"})
              </span>
            </p>
            <p className="mt-1">
              model size: teacher{" "}
              <span className="font-medium text-zinc-900">
                {formatBytes({ value: distillComparison.teacherModelSizeBytes })}
              </span>{" "}
              | student{" "}
              <span className="font-medium text-zinc-900">
                {formatBytes({ value: distillComparison.studentModelSizeBytes })}
              </span>
            </p>
            <p className="mt-1">
              size saved:{" "}
              <span className="font-medium text-zinc-900">
                {formatBytes({ value: distillComparison.sizeSavedBytes })}
              </span>{" "}
              <span className="text-zinc-500">{sizeSavedLabel}</span>
            </p>
            <p className="mt-1">
              params: teacher{" "}
              <span className="font-medium text-zinc-900">
                {formatInt({ value: distillComparison.teacherParamCount })}
              </span>{" "}
              | student{" "}
              <span className="font-medium text-zinc-900">
                {formatInt({ value: distillComparison.studentParamCount })}
              </span>
            </p>
            <p className="mt-1">
              params saved:{" "}
              <span className="font-medium text-zinc-900">
                {formatInt({ value: distillComparison.paramSavedCount })}
              </span>{" "}
              <span className="text-zinc-500">{paramsSavedLabel}</span>
            </p>
            <div className="mt-2 rounded-md border border-zinc-200 bg-white p-2 text-zinc-600">
              <p className="font-medium text-zinc-700">Parameter Math</p>
              <p className="mt-1">
                D = input feature columns: columns of the dataset. Categorical columns are expanded via one-hot encoding.
              </p>
              <p className="mt-1">
                H = hidden dim, L = hidden layers, C = output classes/targets.
              </p>
            </div>
            <p className="mt-2 break-words text-zinc-600">
              Teacher:{" "}
              (D={distillComparison.teacherInputDim ?? "n/a"}, H={distillComparison.teacherHiddenDim ?? "n/a"}, L={distillComparison.teacherNumHiddenLayers ?? "n/a"}, C={distillComparison.teacherOutputDim ?? "n/a"});{" "}
              total params = (D*H + H) + ((L-1)*(H*H + H)) + (H*C + C) + (2*H*L){" = "}
              <span className="font-medium text-zinc-900">
                {formatInt({ value: distillComparison.teacherParamCount })}
              </span>
            </p>
            <p className="mt-1 break-words text-zinc-600">
              Student:{" "}
              (D={distillComparison.studentInputDim ?? "n/a"}, H={distillComparison.studentHiddenDim ?? "n/a"}, L={distillComparison.studentNumHiddenLayers ?? "n/a"}, C={distillComparison.studentOutputDim ?? "n/a"});{" "}
              total params = (D*H + H) + ((L-1)*(H*H + H)) + (H*C + C) + (2*H*L){" = "}
              <span className="font-medium text-zinc-900">
                {formatInt({ value: distillComparison.studentParamCount })}
              </span>
            </p>
          </div>
        ) : null}
        {showModelArtifacts ? (
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
