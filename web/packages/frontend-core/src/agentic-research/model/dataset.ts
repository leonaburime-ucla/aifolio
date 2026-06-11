import type { ApplyDatasetLoadResetInput } from "@aifolio/contracts/entities/agentic-research";

/**
 * Reset stale derived dataset/chart state before loading a new dataset payload.
 *
 * @param input - Required reset dependencies.
 * @returns void
 */
export function applyDatasetLoadReset(input: ApplyDatasetLoadResetInput): void {
  input.actions.setTableRows([]);
  input.actions.setTableColumns([]);
  input.actions.setNumericMatrix([]);
  input.actions.setFeatureNames([]);
  input.actions.setPcaChartSpec(null);
}
