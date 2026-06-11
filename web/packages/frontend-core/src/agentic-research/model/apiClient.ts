import type {
  AgenticResearchApiOptions,
  DatasetManifestEntry,
  DatasetRowsResponse,
  FetchAgenticDatasetManifestInput,
  FetchAgenticDatasetRowsInput,
  FetchAgenticSklearnToolsInput,
  FetchPcaChartSpecPayload,
  FetchPcaChartSpecResult,
  PcaToolResponse,
} from "@aifolio/contracts/entities/agentic-research";
import { buildPcaChartSpec } from "./chart";

type ResolvedAgenticResearchApiRuntimeDeps = {
  fetchImpl: typeof fetch;
  resolveBaseUrl: () => string;
};

function resolveRuntimeDeps(
  options?: AgenticResearchApiOptions
): ResolvedAgenticResearchApiRuntimeDeps {
  const runtimeDeps = options?.runtimeDeps;
  const rawFetchImpl = runtimeDeps?.fetchImpl ?? globalThis.fetch;
  const fetchImpl: typeof fetch = (input, init) =>
    init === undefined ? rawFetchImpl(input) : rawFetchImpl(input, init);
  return {
    fetchImpl,
    resolveBaseUrl: runtimeDeps?.resolveBaseUrl ?? (() => ""),
  };
}

/**
 * Fetches and maps backend ML dataset metadata into agentic research manifest entries.
 *
 * @param _input - Required input object, empty by design for API-shape consistency.
 * @param options - Optional runtime dependencies for fetch and base URL resolution.
 * @returns Dataset manifest entries used by agentic research views.
 * @complexity O(n) over returned datasets, excluding network latency.
 * @overallScore 100
 */
export async function fetchAgenticDatasetManifest(
  _input: FetchAgenticDatasetManifestInput = {},
  options?: AgenticResearchApiOptions
): Promise<DatasetManifestEntry[]> {
  const runtime = resolveRuntimeDeps(options);
  const response = await runtime.fetchImpl(`${runtime.resolveBaseUrl()}/ml-data`);
  if (!response.ok) {
    throw new Error("Failed to load dataset manifest.");
  }
  const payload = (await response.json()) as {
    datasets?: { id: string; label?: string; format?: string }[];
  };

  return (payload.datasets ?? []).map((entry) => ({
    id: entry.id,
    label: entry.label ?? entry.id,
    description: entry.format
      ? `${entry.format.toUpperCase()} dataset from backend/data/ml_data`
      : "Dataset from backend/data/ml_data",
  }));
}

/**
 * Fetches sklearn tool identifiers available to the agentic research UI.
 *
 * @param _input - Required input object, empty by design for API-shape consistency.
 * @param options - Optional runtime dependencies for fetch and base URL resolution.
 * @returns Tool identifiers in backend order.
 * @complexity O(n) over returned tools, excluding network latency.
 * @overallScore 100
 */
export async function fetchAgenticSklearnTools(
  _input: FetchAgenticSklearnToolsInput = {},
  options?: AgenticResearchApiOptions
): Promise<string[]> {
  const runtime = resolveRuntimeDeps(options);
  const response = await runtime.fetchImpl(`${runtime.resolveBaseUrl()}/sklearn-tools`);
  if (!response.ok) {
    throw new Error("Failed to load sklearn tools.");
  }
  const payload = (await response.json()) as { tools?: string[] };
  return payload.tools ?? [];
}

/**
 * Fetches rows and column metadata for a selected dataset id.
 *
 * @param input - Dataset id to load.
 * @param options - Optional runtime dependencies for fetch and base URL resolution.
 * @returns Dataset rows response from the backend.
 * @complexity O(n) over returned row payload size, excluding network latency.
 * @overallScore 100
 */
export async function fetchAgenticDatasetRows(
  input: FetchAgenticDatasetRowsInput,
  options?: AgenticResearchApiOptions
): Promise<DatasetRowsResponse> {
  const runtime = resolveRuntimeDeps(options);
  const response = await runtime.fetchImpl(
    `${runtime.resolveBaseUrl()}/ml-data/${encodeURIComponent(input.datasetId)}`
  );
  if (!response.ok) {
    throw new Error("Failed to load dataset file.");
  }
  return (await response.json()) as DatasetRowsResponse;
}

/**
 * Runs the PCA backend tool and converts the response into a chart spec.
 *
 * @param payload - PCA request payload including matrix, features, and dataset metadata.
 * @param options - Optional runtime dependencies for fetch and base URL resolution.
 * @returns Chart spec for PCA output, or null when the backend returns no usable result.
 * @complexity O(n) over transformed PCA rows, excluding network latency.
 * @overallScore 100
 */
export async function fetchAgenticPcaChartSpec(
  payload: FetchPcaChartSpecPayload,
  options?: AgenticResearchApiOptions
): Promise<FetchPcaChartSpecResult> {
  const runtime = resolveRuntimeDeps(options);
  const response = await runtime.fetchImpl(`${runtime.resolveBaseUrl()}/llm/ds`, {
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
