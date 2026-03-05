import type { ApplyDatasetLoadResetInput } from "@/features/agentic-research/__types__/typescript/logic/agenticResearchDataset.types";

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
