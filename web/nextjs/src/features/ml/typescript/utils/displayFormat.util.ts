/**
 * Formats a byte size into a human-readable unit string.
 * @param params - Required parameters.
 * @param params.value - Byte count value.
 * @returns Formatted size or `n/a` when value is invalid.
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
 * Formats an integer-like numeric value with locale separators.
 * @param params - Required parameters.
 * @param params.value - Integer value to format.
 * @returns Formatted integer or `n/a` when value is invalid.
 */
export function formatInt({
  value,
}: {
  value: number | null | undefined;
}): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return value.toLocaleString();
}
