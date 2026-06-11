import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { formatMetricNumber } from '@aifolio/frontend-core/ml-training';
import { ModalComponent } from '../../../shared/components/modal.component';

@Component({
  selector: 'app-optimal-params-modal',
  imports: [CommonModule, ModalComponent],
  template: `
    <app-modal [isOpen]="isOpen" title="Bayesian Optimization Suggestion" (close)="close()">
      <div class="stack">
        <p class="muted">
          Suggested next hyperparameters for
          <strong>{{ activeAlgorithm || 'this architecture' }}</strong>
          based on completed runs.
        </p>
        <div class="two-grid">
          <p>epochs: <strong>{{ pendingOptimalParams?.epochs ?? 'n/a' }}</strong></p>
          <p>learning_rate: <strong>{{ precise(pendingOptimalParams?.learning_rate, 6) }}</strong></p>
          <p>test_size: <strong>{{ precise(pendingOptimalParams?.test_size, 4) }}</strong></p>
          <p>batch_size: <strong>{{ pendingOptimalParams?.batch_size ?? 'n/a' }}</strong></p>
          <p>hidden_dim: <strong>{{ pendingOptimalParams?.hidden_dim ?? 'n/a' }}</strong></p>
          <p>num_hidden_layers: <strong>{{ pendingOptimalParams?.num_hidden_layers ?? 'n/a' }}</strong></p>
          <p>dropout: <strong>{{ precise(pendingOptimalParams?.dropout, 4) }}</strong></p>
        </div>
        @if (pendingOptimalPrediction) {
          <p class="strong-red">
            Predicted: {{ pendingOptimalPrediction.metricName }} ≈
            {{ formatMetricNumber({ value: pendingOptimalPrediction.metricValue }) }}
          </p>
        }
        <div class="button-row" style="justify-content: flex-end;">
          <button type="button" class="btn secondary" (click)="close()">Cancel</button>
          <button type="button" class="btn primary" [disabled]="!pendingOptimalParams" (click)="apply()">Update Table With Values</button>
        </div>
      </div>
    </app-modal>
  `
})
export class OptimalParamsModalComponent {
  @Input() isOpen = false;
  @Input() pendingOptimalParams: any = null;
  @Input() pendingOptimalPrediction: any = null;
  @Input() activeAlgorithm = '';
  @Input() close: () => void = () => {};
  @Input() apply: () => void = () => {};

  readonly formatMetricNumber = formatMetricNumber;

  precise(value: unknown, precision: number): string {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? String(Number(numeric.toPrecision(precision))) : 'n/a';
  }
}
