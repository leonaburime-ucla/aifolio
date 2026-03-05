import {
  fetchDatasetManifest,
  fetchDatasetRows,
  fetchPcaChartSpec,
  fetchSklearnTools,
} from "@/features/agentic-research/typescript/api/agenticResearchApi";
import type { AgenticResearchApiDeps } from "@/features/agentic-research/__types__/typescript/agenticResearch.types";
import type {
  CreateAgenticResearchApiAdapterInput,
  CreateAgenticResearchApiAdapterOptions,
} from "@/features/agentic-research/__types__/typescript/api/agenticResearchApiAdapter.types";

/**
 * Build Agentic Research API dependencies behind a stable adapter contract.
 *
 * @param _input - Required input object for signature consistency.
 * @param options - Optional transport overrides.
 * @returns API dependency object for orchestrator wiring.
 */
export function createAgenticResearchApiAdapter(
  _input: CreateAgenticResearchApiAdapterInput,
  options: CreateAgenticResearchApiAdapterOptions = {}
): AgenticResearchApiDeps {
  return {
    fetchDatasetManifest: options.fetchDatasetManifest ?? fetchDatasetManifest,
    fetchSklearnTools: options.fetchSklearnTools ?? fetchSklearnTools,
    fetchDatasetRows: options.fetchDatasetRows ?? fetchDatasetRows,
    fetchPcaChartSpec: options.fetchPcaChartSpec ?? fetchPcaChartSpec,
  };
}
