/**
 * Spec: recharts.errors.spec.ts
 * Version: 1.0.0
 */
export const RECHARTS_ERRORS_SPEC_VERSION = "1.0.0";

export const rechartsErrorsSpec = {
  id: "recharts.errors",
  version: RECHARTS_ERRORS_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  registry: [
    {
      code: "UNSUPPORTED_CHART_TYPE",
      source: "ChartRenderer",
      behavior: "render fallback block with unsupported type message",
      retryable: false,
    },
  ],
  mappingRules: [
    "Unsupported chart types degrade to fallback UI; renderer does not throw for unsupported known types.",
  ],
} as const;
