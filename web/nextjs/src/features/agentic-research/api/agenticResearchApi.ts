import type { ChartSpec } from "@/features/ai/types/chart.types";
import type { DatasetManifestEntry } from "@/features/agentic-research/types/agenticResearch.types";
import { getAiApiBaseUrl } from "@/core/config/aiApi";

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

type DatasetRowsResponse = {
  rows?: Array<Record<string, string | number | null>>;
  columns?: string[];
};

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

/**
 * Loads dataset manifest metadata for agentic research datasets.
 */
export async function fetchDatasetManifest(): Promise<DatasetManifestEntry[]> {
  const baseUrl = getAiApiBaseUrl();
  const response = await fetch(`${baseUrl}/ml-data`);
  if (!response.ok) {
    throw new Error("Failed to load dataset manifest.");
  }
  const payload = (await response.json()) as { datasets?: { id: string, label?: string, format?: string }[] };

  // Map the new ml-data schema to the expected agentic research schema
  return (payload.datasets ?? []).map((entry) => ({
    id: entry.id,
    label: entry.label ?? entry.id,
    description: entry.format
      ? `${entry.format.toUpperCase()} dataset from ai/ml/data`
      : "Dataset from ai/ml/data",
  }));
}

/**
 * Loads the list of sklearn tools available to the agentic research UI.
 */
export async function fetchSklearnTools(): Promise<string[]> {
  const baseUrl = getAiApiBaseUrl();
  const response = await fetch(`${baseUrl}/sklearn-tools`);
  if (!response.ok) {
    throw new Error("Failed to load sklearn tools.");
  }
  const payload = (await response.json()) as { tools?: string[] };
  return payload.tools ?? [];
}

/**
 * Loads dataset rows and optional columns for the selected dataset.
 */
export async function fetchDatasetRows(
  datasetId: string
): Promise<DatasetRowsResponse> {
  const baseUrl = getAiApiBaseUrl();
  const response = await fetch(`${baseUrl}/ml-data/${encodeURIComponent(datasetId)}`);
  if (!response.ok) {
    throw new Error("Failed to load dataset file.");
  }
  return (await response.json()) as DatasetRowsResponse;
}

export async function fetchPcaChartSpec(payload: {
  data: number[][];
  feature_names?: string[];
  n_components?: number;
  dataset_id?: string;
  dataset_meta?: Record<string, unknown>;
}): Promise<ChartSpec | null> {
  const baseUrl = getAiApiBaseUrl();
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
