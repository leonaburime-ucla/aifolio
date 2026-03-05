import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";
import type {
  ResolveActiveChartSpecInput,
  ResolveChartToolNameInput,
  ResolveChartToolNameOptions,
} from "@/features/agentic-research/__types__/typescript/logic/agenticResearchChart.types";

export const DEFAULT_TOOL_ACRONYMS: Record<string, string> = {
  pca: "PCA",
  svd: "SVD",
  ica: "ICA",
  nmf: "NMF",
  tsne: "t-SNE",
  knn: "KNN",
  rfe: "RFE",
  rfecv: "RFECV",
  svr: "SVR",
  svc: "SVC",
  lda: "LDA",
  qda: "QDA",
  gmm: "GMM",
  kmeans: "K-Means",
  minibatch: "Mini-Batch",
  dbscan: "DBSCAN",
  optics: "OPTICS",
  pls: "PLS",
  elasticnet: "ElasticNet",
  minmax: "MinMax",
};

/**
 * Resolve active chart with deterministic precedence.
 *
 * @param input - Required chart precedence input.
 * @returns Active chart spec by precedence or null.
 */
export function resolveActiveChartSpec(
  input: ResolveActiveChartSpecInput
): ChartSpec | null {
  return input.pcaChartSpec ?? input.chartSpecs[0] ?? null;
}

/**
 * Format sklearn tool identifier into UI label.
 *
 * @param input - Required tool formatting input.
 * @param options - Optional acronym mapping override.
 * @returns Human-readable tool label.
 */
export function formatToolName(
  input: ResolveChartToolNameInput,
  options?: ResolveChartToolNameOptions
): string {
  const acronyms = options?.acronyms ?? DEFAULT_TOOL_ACRONYMS;
  return input.name
    .split("_")
    .map((word) => acronyms[word] ?? word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
