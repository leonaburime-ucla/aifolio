export type TrainingRunRow = Record<string, string | number | null>;

export type TrainingMetrics = {
  task?: string;
  train_loss?: number;
  test_loss?: number;
  test_metric_name?: string;
  test_metric_value?: number;
};

export type DistillComparison = {
  metricName: string;
  teacherMetricValue: number | null;
  studentMetricValue: number | null;
  qualityDelta: number | null;
  higherIsBetter: boolean;
  teacherTrainingMode: string | null;
  studentTrainingMode: string | null;
  teacherHiddenDim: number | null;
  studentHiddenDim: number | null;
  teacherNumHiddenLayers: number | null;
  studentNumHiddenLayers: number | null;
  teacherInputDim: number | null;
  studentInputDim: number | null;
  teacherOutputDim: number | null;
  studentOutputDim: number | null;
  teacherModelSizeBytes: number | null;
  studentModelSizeBytes: number | null;
  sizeSavedBytes: number | null;
  sizeSavedPercent: number | null;
  teacherParamCount: number | null;
  studentParamCount: number | null;
  paramSavedCount: number | null;
  paramSavedPercent: number | null;
};

export const TRAINING_RUN_COLUMNS = [
  "completed_at",
  "distill_action",
  // Display name for backend `test_metric_value` (e.g. accuracy, rmse).
  "metric_score",
  "train_loss",
  "metric_name",
  "test_loss",
  "result",
  "epochs",
  "learning_rate",
  "test_size",
  "batch_size",
  "hidden_dim",
  "num_hidden_layers",
  "dropout",
  "task",
  "training_mode",
  "target_column",
  "dataset_id",
  "model_id",
  "model_path",
  "error",
] as const;

export function formatCompletedAt(date = new Date()): string {
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  const yy = String(date.getFullYear()).slice(-2);
  const hh = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  const sec = String(date.getSeconds()).padStart(2, "0");
  return `${mm}/${dd}/${yy} ${hh}:${min}:${sec}`;
}

export function formatMetricNumber(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  if (value === 0) return "0";
  const abs = Math.abs(value);

  if (abs < 1e-5) {
    const [mantissa, exponent] = value.toExponential(4).split("e");
    const cleanedExponent = exponent.replace("+", "");
    return `${mantissa}x10^${cleanedExponent}`;
  }

  if (abs >= 1e6) {
    return value.toExponential(4).replace("e", "x10^");
  }

  return Number(value.toPrecision(5)).toString();
}

export function calcTrainingTableHeight(rowsCount: number): number {
  const rowHeight = 48;
  const headerHeight = 64;
  const minHeight = 140;
  const maxHeight = 360;
  const computed = headerHeight + rowsCount * rowHeight;
  return Math.max(minHeight, Math.min(maxHeight, computed));
}
