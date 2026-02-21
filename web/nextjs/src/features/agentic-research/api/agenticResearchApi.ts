import type { ChartSpec } from "@/features/ai/types/chart.types";

type PcaToolResult = {
  transformed?: number[][];
  explained_variance_ratio?: number[];
  feature_importance?: Array<{ feature: string; importance: number }>;
  feature_names?: string[];
};

type PcaToolResponse = {
  status?: string;
  mode?: string;
  result?: PcaToolResult | null;
};

const DEFAULT_AI_API_URL = "http://127.0.0.1:8000";

function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_AI_API_URL || DEFAULT_AI_API_URL;
}

function buildPcaChartSpec(result: PcaToolResult): ChartSpec | null {
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

export async function fetchPcaChartSpec(payload: {
  data: number[][];
  feature_names?: string[];
  n_components?: number;
  dataset_id?: string;
  dataset_meta?: Record<string, unknown>;
}): Promise<ChartSpec | null> {
  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/llm/ds`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: "Run PCA and return the transformed points.",
      tool_name: "pca_transform",
      tool_args: {
        data: payload.data,
        n_components: payload.n_components ?? 2,
        feature_names: payload.feature_names,
        dataset_id: payload.dataset_id,
        dataset_meta: payload.dataset_meta,
      },
    }),
  });

  if (!response.ok) return null;
  const data = (await response.json()) as PcaToolResponse;
  if (!data?.result) return null;
  return buildPcaChartSpec(data.result);
}
