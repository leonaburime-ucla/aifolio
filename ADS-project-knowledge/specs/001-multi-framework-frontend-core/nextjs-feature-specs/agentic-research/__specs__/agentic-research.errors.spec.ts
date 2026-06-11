/**
 * Spec: agentic-research.errors.spec.ts
 * Version: 1.1.0
 */
export const AGENTIC_RESEARCH_ERRORS_SPEC_VERSION = "1.1.0";

export const agenticResearchErrorsSpec = {
  id: "agentic-research.errors",
  version: AGENTIC_RESEARCH_ERRORS_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  registry: [
    {
      code: "CHART_NOT_FOUND",
      source: "ar-remove_chart_spec",
      retryable: true,
      payload: ["chart_id", "available_chart_ids"],
    },
    {
      code: "INDEX_OUT_OF_RANGE",
      source: "ar-reorder_chart_specs",
      retryable: true,
      payload: ["from_index", "to_index", "chart_count"],
    },
    {
      code: "INVALID_REORDER_PAYLOAD",
      source: "ar-reorder_chart_specs",
      retryable: true,
      payload: ["hint"],
    },
    {
      code: "INVALID_DATASET_ID",
      source: "ar-set_active_dataset",
      retryable: true,
      payload: ["dataset_id", "allowed_dataset_ids"],
    },
  ],
  mappingRules: [
    "Manifest/tool/dataset fetch errors map to user-safe state error strings.",
    "Dataset load failures clear stale table/chart derived state.",
    "PCA API non-OK/missing-result maps to null chart result (non-throw path).",
  ],
} as const;
