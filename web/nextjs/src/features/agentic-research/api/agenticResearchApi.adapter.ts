// Stays in Next.js app: wires app-local API functions (which bind process.env base URL)
// into the adapter shape consumed by React hooks — deployment + React coupling.
import {
  fetchDatasetManifest,
  fetchDatasetRows,
  fetchPcaChartSpec,
  fetchSklearnTools,
} from "@/features/agentic-research/api/agenticResearchApi";
import type {
  AgenticResearchApiDeps,
  CreateAgenticResearchApiAdapterInput,
  CreateAgenticResearchApiAdapterOptions,
} from "@aifolio/contracts/entities/agentic-research";

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
