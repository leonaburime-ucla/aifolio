import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { ChartStoreService } from '../../../shared/state/chart-store.service';
import { ChartRendererComponent } from './chart-renderer.component';

@Component({
  selector: 'app-charts-workspace',
  imports: [CommonModule, ChartRendererComponent],
  template: `
    <div class="stack">
      <details class="details-card" open>
        <summary>How to Use Page + Prompts to Try</summary>
        <div class="stack-tight" style="margin-top: .75rem;">
          <p>This AI Chat creates charts from internal sample data. It does not use real-time APIs; chart data comes from backend LLM/tool responses.</p>
          <p><strong>Sample prompts to try:</strong></p>
          <ul>
            <li>Create a line chart of Solana and Bitcoin for the past 5 months.</li>
            <li>Create an area chart of Peruvian beef exports over the past 15 years.</li>
            <li>Show Manhattan vs London vs Paris average rent since 2000 as a share of salary.</li>
            <li>Plot US debt levels for the past 50 years and estimate the next 20 years.</li>
            <li>Make a scatter chart comparing Bitcoin and Ethereum returns over the last 30 days.</li>
          </ul>
        </div>
      </details>

      @if (chartStore.chartSpecs().length === 0) {
        <div class="empty-panel">Charts generated from chat will appear here.</div>
      } @else {
        <div class="stack">
          @for (spec of chartStore.chartSpecs(); track spec.id) {
            <app-chart-renderer [spec]="spec" [removable]="true" (remove)="chartStore.removeChartSpec($event)" />
          }
        </div>
      }
    </div>
  `
})
export class ChartsWorkspaceComponent {
  readonly chartStore = inject(ChartStoreService);
}
