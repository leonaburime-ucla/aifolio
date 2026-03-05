import type {
  ResolveDefaultDatasetIdInput,
  ToDatasetOptionsInput,
  ToDatasetOptionsResult,
} from "@/features/agentic-research/__types__/typescript/logic/agenticResearchManifest.types";

/**
 * Resolve the default selected dataset using deterministic fallback order.
 *
 * @param input - Required dataset selection inputs.
 * @returns Selected dataset id using existing selection or first dataset fallback.
 */
export function resolveDefaultDatasetId(
  input: ResolveDefaultDatasetIdInput
): string | null {
  return input.selectedDatasetId ?? input.datasets[0]?.id ?? null;
}

/**
 * Map manifest entries into deterministic dataset combobox options.
 *
 * @param input - Required manifest mapping input.
 * @returns Dataset options preserving manifest order.
 */
export function toDatasetOptions(
  input: ToDatasetOptionsInput
): ToDatasetOptionsResult {
  return input.datasetManifest.map((entry) => ({
    id: entry.id,
    label: entry.label,
    description: entry.description,
  }));
}
