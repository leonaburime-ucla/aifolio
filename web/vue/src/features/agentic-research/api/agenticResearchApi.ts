import {
  fetchAgenticDatasetManifest,
  fetchAgenticSklearnTools,
  fetchAgenticDatasetRows,
} from "@aifolio/frontend-core/agentic-research";
import type { DatasetRowsResponse } from "@aifolio/contracts/entities/agentic-research";

type ApiOptions = { baseUrl: string };

export function createAgenticResearchApi({ baseUrl }: ApiOptions) {
  const runtimeDeps = { resolveBaseUrl: () => baseUrl };

  async function loadManifest() {
    return fetchAgenticDatasetManifest({}, { runtimeDeps });
  }

  async function loadTools() {
    return fetchAgenticSklearnTools({}, { runtimeDeps });
  }

  async function loadDataset(datasetId: string): Promise<DatasetRowsResponse> {
    return fetchAgenticDatasetRows({ datasetId }, { runtimeDeps });
  }

  return { loadManifest, loadTools, loadDataset };
}
