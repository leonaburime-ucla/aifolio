/**
 * Spec: agentic-research.api.spec.ts
 * Version: 1.1.0
 */
export const AGENTIC_RESEARCH_API_SPEC_VERSION = "1.1.0";

export const agenticResearchApiSpec = {
  id: "agentic-research.api",
  version: AGENTIC_RESEARCH_API_SPEC_VERSION,
  status: "draft",
  lastUpdated: "2026-02-23",
  contracts: {
    fetchDatasetManifest: {
      endpoint: "GET /sample-data",
      response: {
        datasets: "Array<{id:string,label:string,description?:string,...}>",
      },
      onFailure: "throw Error('Failed to load dataset manifest.')",
    },
    fetchSklearnTools: {
      endpoint: "GET /sklearn-tools",
      response: {
        tools: "string[]",
      },
      onFailure: "throw Error('Failed to load sklearn tools.')",
    },
    fetchDatasetRows: {
      endpoint: "GET /sample-data/:id",
      response: {
        rows: "Array<Record<string,string|number|null>>",
        columns: "string[] (optional)",
      },
      onFailure: "throw Error('Failed to load dataset file.')",
    },
    fetchPcaChartSpec: {
      endpoint: "POST /llm/ds",
      request: {
        message: "Run PCA and return the transformed points.",
        tool_name: "pca_transform",
        tool_args: "{data:number[][],n_components?:number,feature_names?:string[],dataset_id?:string,dataset_meta?:Record<string,unknown>}",
      },
      response: "ChartSpec | null",
      onFailure: "return null for non-OK or missing result payload",
    },
  },
  deterministicRules: [
    "Manifest and dataset rows APIs throw user-safe errors on non-OK responses.",
    "PCA API returns null rather than throw for non-OK/missing-result responses.",
  ],
} as const;
