/**
 * Formats an optional percentage value into parenthesized display copy.
 *
 * @param params - Value to format and fallback copy for missing values.
 * @returns Formatted percent label or fallback copy.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function formatPercentLabel(
  {
    value,
    fallback,
  }: {
    value: number | null | undefined;
    fallback: string;
  },
  {}: Record<string, never> = {}
): string {
  if (typeof value !== "number" || Number.isNaN(value)) return fallback;
  return `(${Number(value.toFixed(2))}%)`;
}

/**
 * Resolves whether model artifact details should be rendered.
 *
 * @param params - Optional model id and model path.
 * @returns True when either artifact identifier is present.
 * @complexity O(1) time and space.
 * @overallScore 100
 */
export function hasModelArtifacts(
  {
    modelId,
    modelPath,
  }: {
    modelId: string | null | undefined;
    modelPath: string | null | undefined;
  },
  {}: Record<string, never> = {}
): boolean {
  return Boolean(modelId || modelPath);
}
