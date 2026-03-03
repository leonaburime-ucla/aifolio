export type {
  DistillComparison,
  TrainingMetrics,
  TrainingRunRow,
} from "@/features/ml/__types__/typescript/utils/trainingRuns.types";
type EmptyOptions = Record<string, never>;

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

/**
 * Formats a completion timestamp for the training runs table.
 *
 * @param params - Required parameter object.
 * @param params.date - Source date to format. Defaults to the current time.
 * @param _options - Optional reserved options object.
 * @returns A `MM/DD/YY HH:mm:ss` formatted string.
 */
export function formatCompletedAt(
  { date = new Date() }: { date?: Date },
  {}: EmptyOptions = {}
): string {
  const mm = String(date.getMonth() + 1).padStart(2, "0");
  const dd = String(date.getDate()).padStart(2, "0");
  const yy = String(date.getFullYear()).slice(-2);
  const hh = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  const sec = String(date.getSeconds()).padStart(2, "0");
  return `${mm}/${dd}/${yy} ${hh}:${min}:${sec}`;
}

/**
 * Formats numeric metrics for compact UI display.
 *
 * @param params - Required parameter object.
 * @param params.value - Raw metric value.
 * @param _options - Optional reserved options object.
 * @returns A human-readable number string or `n/a`.
 */
export function formatMetricNumber(
  { value }: { value: unknown },
  {}: EmptyOptions = {}
): string {
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

/**
 * Calculates bounded table height based on row count.
 *
 * @param params - Required parameter object.
 * @param params.rowsCount - Number of rows currently rendered.
 * @param _options - Optional reserved options object.
 * @returns Pixel height clamped to feature min/max constraints.
 */
export function calcTrainingTableHeight(
  { rowsCount }: { rowsCount: number },
  {}: EmptyOptions = {}
): number {
  const rowHeight = 48;
  const headerHeight = 64;
  const minHeight = 140;
  const maxHeight = 360;
  const computed = headerHeight + rowsCount * rowHeight;
  return Math.max(minHeight, Math.min(maxHeight, computed));
}
