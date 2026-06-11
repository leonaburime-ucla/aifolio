// Stays in Next.js app: binds getAiApiBaseUrl (process.env / browser proxy) as the
// default base URL for all agentic-research API calls — deployment-specific wiring.
import type {
  DatasetManifestEntry,
  DatasetRowsResponse,
  FetchPcaChartSpecPayload,
  FetchPcaChartSpecResult,
} from "@aifolio/contracts/entities/agentic-research";
import {
  fetchAgenticDatasetManifest,
  fetchAgenticDatasetRows,
  fetchAgenticPcaChartSpec,
  fetchAgenticSklearnTools,
} from "@aifolio/frontend-core/agentic-research";
import { getAiApiBaseUrl } from "@/core/config/aiApi";

const runtimeOptions = {
  runtimeDeps: {
    resolveBaseUrl: getAiApiBaseUrl,
  },
};

/**
 * Loads dataset manifest metadata using the Next app AI API base URL.
 *
 * @returns Agentic research dataset manifest entries.
 * @complexity O(n) over returned datasets, excluding network latency.
 * @overallScore 100
 */
export async function fetchDatasetManifest(): Promise<DatasetManifestEntry[]> {
  return fetchAgenticDatasetManifest({}, runtimeOptions);
}

/**
 * Loads sklearn tool identifiers using the Next app AI API base URL.
 *
 * @returns Tool identifiers in backend order.
 * @complexity O(n) over returned tools, excluding network latency.
 * @overallScore 100
 */
export async function fetchSklearnTools(): Promise<string[]> {
  return fetchAgenticSklearnTools({}, runtimeOptions);
}

/**
 * Loads dataset rows and optional columns for the selected dataset.
 *
 * @param datasetId - Dataset id to load.
 * @returns Dataset rows response from the backend.
 * @complexity O(n) over returned row payload size, excluding network latency.
 * @overallScore 100
 */
export async function fetchDatasetRows(
  datasetId: string
): Promise<DatasetRowsResponse> {
  return fetchAgenticDatasetRows({ datasetId }, runtimeOptions);
}

/**
 * Runs the PCA backend tool and converts the response into a chart spec.
 *
 * @param payload - PCA request payload including matrix, features, and dataset metadata.
 * @returns Chart spec for PCA output, or null when the backend returns no usable result.
 * @complexity O(n) over transformed PCA rows, excluding network latency.
 * @overallScore 100
 */
export async function fetchPcaChartSpec(
  payload: FetchPcaChartSpecPayload
): Promise<FetchPcaChartSpecResult> {
  return fetchAgenticPcaChartSpec(payload, runtimeOptions);
}
