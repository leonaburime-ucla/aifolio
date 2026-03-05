/**
 * Formats an optional percentage value into "(N%)" display copy.
 * @param params - Required parameters.
 * @returns Formatted percent label, or fallback copy when value is not numeric.
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
 * Resolves whether model artifact details should be shown.
 * @param params - Required parameters.
 * @returns True when either model id or path is present.
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
