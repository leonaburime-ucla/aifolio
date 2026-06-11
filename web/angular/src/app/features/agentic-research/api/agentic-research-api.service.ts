import { Injectable } from '@angular/core';
import {
  fetchAgenticDatasetManifest,
  fetchAgenticDatasetRows,
  fetchAgenticSklearnTools,
} from '@aifolio/frontend-core/agentic-research';
import type { DatasetRowsResponse } from '@aifolio/contracts/entities/agentic-research';

@Injectable({ providedIn: 'root' })
export class AgenticResearchApiService {
  loadManifest(baseUrl: string) {
    return fetchAgenticDatasetManifest({}, { runtimeDeps: { resolveBaseUrl: () => baseUrl } });
  }

  loadTools(baseUrl: string) {
    return fetchAgenticSklearnTools({}, { runtimeDeps: { resolveBaseUrl: () => baseUrl } });
  }

  loadDataset(baseUrl: string, datasetId: string): Promise<DatasetRowsResponse> {
    return fetchAgenticDatasetRows({ datasetId }, { runtimeDeps: { resolveBaseUrl: () => baseUrl } });
  }
}
