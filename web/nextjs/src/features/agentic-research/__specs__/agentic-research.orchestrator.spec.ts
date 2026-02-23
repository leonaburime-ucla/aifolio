/**
 * Spec: agentic-research.orchestrator.spec.ts
 * Version: 1.1.0
 */
export const AGENTIC_RESEARCH_ORCH_SPEC_VERSION = "1.1.0";

export const agenticResearchOrchestratorSpec = {
  id: "agentic-research.orchestrator",
  version: AGENTIC_RESEARCH_ORCH_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  units: [
    "useAgenticResearchOrchestrator",
    "useAgenticResearchIntegration",
    "useAgenticResearchLogic",
  ],
  inputContract: {
    useStatePort: "UseAgenticResearchStatePort (optional, default adapter)",
    useChartStatePort: "UseAgenticResearchChartStatePort (optional, default adapter)",
  },
  outputContract: {
    required: [
      "datasetOptions",
      "selectedDatasetId",
      "tableRows",
      "tableColumns",
      "chartSpecs",
      "activeChartSpec",
      "groupedTools",
      "formatToolName",
    ],
    rule: "activeChartSpec = pcaChartSpec ?? chartSpecs[0] ?? null",
  },
  behaviorRules: [
    "Manifest load sets selected dataset fallback when missing.",
    "Dataset load runs only when selectedDatasetId exists and manifest is non-empty.",
    "State/API dependencies are injected through ports/deps, not direct page imports.",
  ],
} as const;
