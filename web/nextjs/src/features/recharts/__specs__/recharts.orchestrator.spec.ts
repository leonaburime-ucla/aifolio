/**
 * Spec: recharts.orchestrator.spec.ts
 * Version: 1.1.0
 */
export const RECHARTS_ORCHESTRATOR_SPEC_VERSION = "1.1.0";

export const rechartsOrchestratorSpec = {
  id: "recharts.orchestrator",
  version: RECHARTS_ORCHESTRATOR_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  units: ["useChartOrchestrator"],
  inputContract: {
    useChartManagementPort: "optional injected hook returning {chartSpecs, removeChartSpec}",
  },
  outputContract: {
    required: ["chartSpecs", "removeChartSpec"],
    behavior: "returns injected management port output unchanged",
  },
  rules: [
    "No direct store import in orchestrator body (port injection only).",
    "No import from app/core screen consumers.",
  ],
} as const;
