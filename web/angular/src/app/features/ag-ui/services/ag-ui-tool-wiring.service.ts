import { DestroyRef, inject, Injectable, signal, type WritableSignal } from '@angular/core';
import { AgUiToolRegistryService } from './ag-ui-tool-registry.service';
import { ChartStoreService } from '../../../shared/state/chart-store.service';
import {
  handleAddChartSpec,
  handleSwitchAgUiTab,
} from '@aifolio/frontend-core/ag-ui';
import type { AgUiWorkspaceTab } from '@aifolio/contracts/entities/ag-ui';
import type { ChartSpec } from '@aifolio/contracts/entities/chart';

export type AgUiWiringDeps = {
  activeTab: () => AgUiWorkspaceTab;
  setActiveTab: (tab: AgUiWorkspaceTab) => void;
  activeDatasetId: () => string | null;
  setActiveDatasetId: (id: string) => void;
};

@Injectable({ providedIn: 'root' })
export class AgUiToolWiringService {
  private readonly registry = inject(AgUiToolRegistryService);
  private readonly chartStore = inject(ChartStoreService);
  private wired = false;

  wire(deps: AgUiWiringDeps): void {
    if (this.wired) return;
    this.wired = true;

    this.registry.register('switch_ag_ui_tab', (args) => {
      const tab = args['tab'] as string;
      const result = handleSwitchAgUiTab(tab);
      if (result.status === 'ok') {
        deps.setActiveTab(result.tab);
      }
      return result;
    });

    this.registry.register('add_chart_spec', (args) => {
      const result = handleAddChartSpec(
        { chartSpec: args['chartSpec'], chartSpecs: args['chartSpecs'] as unknown[] },
        (spec: ChartSpec) => this.chartStore.addChartSpec(spec)
      );
      return result;
    });

    this.registry.register('clear_charts', () => {
      this.chartStore.clearCharts();
      return { status: 'ok' };
    });

    this.registry.register('set_active_dataset', (args) => {
      const datasetId = args['dataset_id'] as string;
      if (!datasetId) return { status: 'error', code: 'MISSING_DATASET_ID' };
      deps.setActiveDatasetId(datasetId);
      return { status: 'ok', dataset_id: datasetId };
    });
  }

  unwire(): void {
    if (!this.wired) return;
    this.wired = false;
    this.registry.unregister('switch_ag_ui_tab');
    this.registry.unregister('add_chart_spec');
    this.registry.unregister('clear_charts');
    this.registry.unregister('set_active_dataset');
  }
}
