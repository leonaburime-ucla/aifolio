import type { ChartSpec } from "@aifolio/contracts/entities/chart";
import type {
  PcaToolResult,
  ResolveActiveChartSpecInput,
  ResolveChartToolNameInput,
  ResolveChartToolNameOptions,
} from "@aifolio/contracts/entities/agentic-research";

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

/**
 * Convert PCA tool output into the chart contract consumed by renderers.
 *
 * @param result - PCA tool result payload from the backend.
 * @returns Scatter chart spec for transformed points, or null when there are no points.
 */
export function buildPcaChartSpec(result: PcaToolResult): ChartSpec | null {
  const transformed = result.transformed ?? [];
  if (!transformed.length) return null;

  const points = transformed.map((row, index) => ({
    id: `pca-${index + 1}`,
    pc1: row[0] ?? 0,
    pc2: row[1] ?? 0,
  }));
  const variance = result.explained_variance_ratio ?? [];
  const varianceText =
    variance.length >= 2
      ? `Explained variance: ${variance
        .slice(0, 3)
        .map((value, index) => `PC${index + 1} ${(value * 100).toFixed(1)}%`)
        .join(", ")}`
      : undefined;

  return {
    id: "agentic-research-pca",
    title: "PCA Projection",
    description: varianceText,
    type: "scatter",
    xKey: "pc1",
    yKeys: ["pc2"],
    xLabel: "PC1",
    yLabel: "PC2",
    data: points,
  };
}
