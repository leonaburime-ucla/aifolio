/**
 * Formats byte counts using compact binary units.
 *
 * @param params - Required byte value parameter.
 * @returns Human-readable byte string or `n/a` for invalid values.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function formatBytes({
  value,
}: {
  value: number | null | undefined;
}): string {
  if (typeof value !== "number" || Number.isNaN(value) || value < 0) return "n/a";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / (1024 ** 2)).toFixed(2)} MB`;
  return `${(value / (1024 ** 3)).toFixed(2)} GB`;
}

/**
 * Formats integer-like counts with locale grouping.
 *
 * @param params - Required numeric value parameter.
 * @returns Grouped integer string or `n/a` for invalid values.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function formatInt({
  value,
}: {
  value: number | null | undefined;
}): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return value.toLocaleString();
}

/**
 * Formats a completion timestamp for ML tables.
 *
 * @param params - Optional source date. Defaults to current time.
 * @returns `MM/DD/YY HH:mm:ss` timestamp string.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function formatCompletedAt(
  { date = new Date() }: { date?: Date },
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
 * Formats metric numbers for compact table display.
 *
 * @param params - Required raw metric value.
 * @returns Compact numeric string or `n/a`.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function formatMetricNumber(
  { value }: { value: unknown },
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
 * Calculates bounded training table height from row count.
 *
 * @param params - Required row count parameter.
 * @returns Pixel height clamped to the table min/max range.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function calcTrainingTableHeight(
  { rowsCount }: { rowsCount: number },
): number {
  const rowHeight = 48;
  const headerHeight = 64;
  const minHeight = 140;
  const maxHeight = 360;
  const computed = headerHeight + rowsCount * rowHeight;
  return Math.max(minHeight, Math.min(maxHeight, computed));
}
