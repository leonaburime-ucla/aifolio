import type { AgenticResearchApiDeps } from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

export type CreateAgenticResearchApiAdapterInput = Record<string, never>;

export type CreateAgenticResearchApiAdapterOptions = {
  fetchDatasetManifest?: AgenticResearchApiDeps["fetchDatasetManifest"];
  fetchSklearnTools?: AgenticResearchApiDeps["fetchSklearnTools"];
  fetchDatasetRows?: AgenticResearchApiDeps["fetchDatasetRows"];
  fetchPcaChartSpec?: AgenticResearchApiDeps["fetchPcaChartSpec"];
};
