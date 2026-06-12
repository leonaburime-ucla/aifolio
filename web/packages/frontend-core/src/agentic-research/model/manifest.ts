import type {
  ResolveDefaultDatasetIdInput,
  ToDatasetOptionsInput,
  ToDatasetOptionsResult,
} from "@aifolio/contracts/entities/agentic-research";
import { resolvePreferredDatasetId } from "../../config/mlDatasets";

/**
 * Resolve the default selected dataset using deterministic fallback order.
 *
 * @param input - Required dataset selection inputs.
 * @returns Existing selection, preferred customer churn dataset, first dataset, or null.
 */
export function resolveDefaultDatasetId(
  input: ResolveDefaultDatasetIdInput
): string | null {
  return resolvePreferredDatasetId(input);
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
