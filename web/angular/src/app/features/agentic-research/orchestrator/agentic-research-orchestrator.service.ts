import { Injectable, computed, inject, signal } from '@angular/core';
import { formatToolName, groupSklearnTools, resolveDefaultDatasetId, toDatasetOptions } from '@aifolio/frontend-core/agentic-research';
import type { DatasetOption } from '../../../shared/types/dataset-option';
import { ChartStoreService } from '../../../shared/state/chart-store.service';
import { AgenticResearchApiService } from '../api/agentic-research-api.service';

@Injectable()
export class AgenticResearchOrchestrator {
  private readonly api = inject(AgenticResearchApiService);
  readonly chartStore = inject(ChartStoreService);

  readonly baseUrl = signal('/api/ai');
  readonly datasetOptions = signal<DatasetOption[]>([]);
  readonly selectedDatasetId = signal<string | null>(null);
  readonly tableRows = signal<Record<string, unknown>[]>([]);
  readonly tableColumns = signal<string[]>([]);
  readonly sklearnTools = signal<string[]>([]);
  readonly isLoading = signal(false);
  readonly error = signal<string | null>(null);

  readonly toolGroups = computed(() => {
    const grouped = groupSklearnTools({ tools: this.sklearnTools() });
    return ['Decomposition & Embeddings', 'Classification', 'Clustering', 'Regression']
      .filter((name) => grouped[name]?.length)
      .map((name) => ({
        name,
        formatted: grouped[name].map((tool: string) => formatToolName({ name: tool })).join(', '),
      }));
  });

  async init(onDatasetChange?: (id: string) => void): Promise<void> {
    await Promise.all([this.loadManifest(), this.loadTools()]);
    const selected = this.selectedDatasetId();
    if (selected) {
      onDatasetChange?.(selected);
      await this.loadDataset(selected);
    }
  }

  async loadManifest(): Promise<void> {
    try {
      const entries = await this.api.loadManifest(this.baseUrl());
      this.datasetOptions.set(toDatasetOptions({ datasetManifest: entries }));
      const resolved = resolveDefaultDatasetId({ selectedDatasetId: this.selectedDatasetId(), datasets: entries });
      if (resolved && !this.selectedDatasetId()) this.selectedDatasetId.set(resolved);
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Failed to load manifest.');
    }
  }

  async loadTools(): Promise<void> {
    try {
      this.sklearnTools.set(await this.api.loadTools(this.baseUrl()));
    } catch {
      this.sklearnTools.set([]);
    }
  }

  async loadDataset(id: string): Promise<void> {
    this.isLoading.set(true);
    this.error.set(null);
    this.tableRows.set([]);
    this.tableColumns.set([]);
    try {
      const payload = await this.api.loadDataset(this.baseUrl(), id);
      const rows = payload.rows ?? [];
      this.tableRows.set(rows);
      this.tableColumns.set(payload.columns ?? (rows.length > 0 ? Object.keys(rows[0]) : []));
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : 'Failed to load dataset.');
    } finally {
      this.isLoading.set(false);
    }
  }

  async onDatasetChange(id: string, emit?: (id: string) => void): Promise<void> {
    this.selectedDatasetId.set(id);
    emit?.(id);
    await this.loadDataset(id);
  }
}
