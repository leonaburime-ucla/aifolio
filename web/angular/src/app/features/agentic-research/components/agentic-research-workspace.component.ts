import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, OnInit, Output, inject } from '@angular/core';
import { DataTableComponent } from '../../../shared/components/data-table.component';
import { DatasetComboboxComponent } from '../../../shared/components/dataset-combobox.component';
import { ChartRendererComponent } from '../../recharts/components/chart-renderer.component';
import { AgenticResearchOrchestrator } from '../orchestrator/agentic-research-orchestrator.service';

@Component({
  selector: 'app-agentic-research-workspace',
  imports: [CommonModule, DatasetComboboxComponent, DataTableComponent, ChartRendererComponent],
  providers: [AgenticResearchOrchestrator],
  template: `
    <div class="stack">
      <details class="details-card" open>
        <summary>{{ showPrompts ? 'ML Algorithms + Sample Prompts' : 'ML Algorithms' }}</summary>
        <div style="margin-top: .75rem;">
          @if (showPrompts) {
            <p class="strong-red">Results take 1-2min</p>
            <p><strong>Sample Prompts</strong></p>
            <ol style="margin: .75rem 0 1rem; padding-left: 1.25rem; display: flex; flex-direction: column; gap: .25rem;">
              <li>Run PCA analysis</li>
              <li>Run NMF Decomposition and PLSR</li>
              <li>Change the dataset to fraud detection and run Random Forest</li>
            </ol>
          }
          @if (model.sklearnTools().length > 0) {
            <div class="algo-groups">
              @for (group of model.toolGroups(); track group.name) {
                <div class="algo-group">
                  <p class="algo-group-title">{{ group.name }}</p>
                  <p class="algo-group-items">{{ group.formatted }}</p>
                </div>
              }
            </div>
          } @else {
            <p>Loading...</p>
          }
        </div>
      </details>

      <details class="details-card">
        <summary>Preprocessing Notes</summary>
        <div class="stack-tight" style="margin-top: .75rem;">
          <p><strong>Categorical Encoding:</strong> Text columns with &lt;= 20 unique values are automatically One-Hot Encoded.</p>
          <p><strong>High Cardinality &amp; IDs:</strong> Text columns with &gt; 20 unique values or ID-like names are dropped.</p>
          <p><strong>Date Parsing:</strong> Dates and timestamps are extracted into Year, Month, and Day numeric features.</p>
          <p><strong>Missing Values:</strong> Missing numeric values are imputed using the column median.</p>
          <p><strong>Feature Scaling:</strong> All features are standardized to zero mean and unit variance.</p>
        </div>
      </details>

      <div class="stack-tight">
        <p class="page-kicker">Dataset</p>
        <app-dataset-combobox
          [options]="model.datasetOptions()"
          [selectedId]="model.selectedDatasetId()"
          (changed)="model.onDatasetChange($event, emitDatasetChange)"
        />
      </div>

      <details class="section-card" open>
        <summary style="cursor: pointer; font-weight: 700;">Charts</summary>
        <div style="margin-top: 1rem;">
          @if (model.isLoading()) {
            <div class="empty-panel">Loading dataset...</div>
          } @else if (model.chartStore.chartSpecs().length > 0) {
            <div class="stack">
              @for (spec of model.chartStore.chartSpecs(); track spec.id) {
                <app-chart-renderer
                  [spec]="spec"
                  [removable]="true"
                  (remove)="model.chartStore.removeChartSpec($event)"
                />
              }
            </div>
          } @else {
            <div class="empty-panel">{{ model.error() ?? 'No analysis chart data available yet.' }}</div>
          }
        </div>
      </details>

      <app-data-table [rows]="model.tableRows()" [columns]="model.tableColumns()" />
    </div>
  `
})
export class AgenticResearchWorkspaceComponent implements OnInit {
  @Input() showPrompts = true;
  @Output() datasetChange = new EventEmitter<string>();
  readonly model = inject(AgenticResearchOrchestrator);
  readonly emitDatasetChange = (id: string) => this.datasetChange.emit(id);

  ngOnInit(): void {
    void this.model.init(this.emitDatasetChange);
  }
}
