import { CommonModule } from '@angular/common';
import { Component, Input, computed } from '@angular/core';
import { formatBytes, formatInt, formatMetricNumber, formatPercentLabel, hasModelArtifacts } from '@aifolio/frontend-core/ml-training';
import { ModalComponent } from '../../../shared/components/modal.component';

@Component({
  selector: 'app-distill-metrics-modal',
  imports: [CommonModule, ModalComponent],
  template: `
    <app-modal [isOpen]="isOpen" title="Distillation Metrics" (close)="close()">
      <div class="stack">
        <div class="metrics-grid">
          <div class="section-card">
            <p class="page-kicker" style="margin: 0;">metric_name</p>
            <p style="margin: .5rem 0 0;"><strong>{{ distillMetrics?.test_metric_name ?? 'n/a' }}</strong></p>
          </div>
          <div class="section-card">
            <p class="page-kicker" style="margin: 0;">metric_score</p>
            <p style="margin: .5rem 0 0;"><strong>{{ formatMetricNumber({ value: distillMetrics?.test_metric_value }) }}</strong></p>
          </div>
          <div class="section-card">
            <p class="page-kicker" style="margin: 0;">train_loss</p>
            <p style="margin: .5rem 0 0;"><strong>{{ formatMetricNumber({ value: distillMetrics?.train_loss }) }}</strong></p>
          </div>
          <div class="section-card">
            <p class="page-kicker" style="margin: 0;">test_loss</p>
            <p style="margin: .5rem 0 0;"><strong>{{ formatMetricNumber({ value: distillMetrics?.test_loss }) }}</strong></p>
          </div>
        </div>

        @if (distillComparison) {
          <div class="section-card" style="display: flex; flex-direction: column; gap: .35rem;">
            <p style="margin: 0;"><strong>Teacher vs Student</strong></p>
            <p style="margin: 0;">
              metric ({{ distillComparison.metricName }}): teacher
              <strong>{{ formatMetricNumber({ value: distillComparison.teacherMetricValue }) }}</strong>
              | student
              <strong>{{ formatMetricNumber({ value: distillComparison.studentMetricValue }) }}</strong>
            </p>
            <p style="margin: 0;">
              quality delta (student vs teacher):
              <strong>{{ formatMetricNumber({ value: distillComparison.qualityDelta }) }}</strong>
              <span class="muted">({{ distillComparison.higherIsBetter ? 'higher is better' : 'lower is better' }})</span>
            </p>
            <p style="margin: 0;">
              model size: teacher <strong>{{ formatBytes({ value: distillComparison.teacherModelSizeBytes }) }}</strong>
              | student <strong>{{ formatBytes({ value: distillComparison.studentModelSizeBytes }) }}</strong>
            </p>
            <p style="margin: 0;">size saved: <strong>{{ formatBytes({ value: distillComparison.sizeSavedBytes }) }}</strong><span class="muted">{{ sizeSavedLabel() }}</span></p>
            <p style="margin: 0;">
              params: teacher <strong>{{ formatInt({ value: distillComparison.teacherParamCount }) }}</strong>
              | student <strong>{{ formatInt({ value: distillComparison.studentParamCount }) }}</strong>
            </p>
            <p style="margin: 0;">params saved: <strong>{{ formatInt({ value: distillComparison.paramSavedCount }) }}</strong><span class="muted">{{ paramsSavedLabel() }}</span></p>
            <div class="section-card" style="margin-top: .5rem; background: #fafafa;">
              <p style="margin: 0;"><strong>Parameter Math</strong></p>
              <p style="margin: .25rem 0 0;">D = input feature columns: columns of the dataset. Categorical columns are expanded via one-hot encoding.</p>
              <p style="margin: .25rem 0 0;">H = hidden dim, L = hidden layers, C = output classes/targets.</p>
            </div>
            @if (distillComparison.teacherParamFormula) {
              <p style="margin: .5rem 0 0;">Teacher: {{ distillComparison.teacherParamFormula }}</p>
            }
            @if (distillComparison.studentParamFormula) {
              <p style="margin: 0;">Student: {{ distillComparison.studentParamFormula }}</p>
            }
          </div>
        }

        @if (showModelArtifacts()) {
          <div class="section-card">
            <p style="margin: 0;">model_id: <strong>{{ distillModelId ?? 'n/a' }}</strong></p>
            <p style="margin: .25rem 0 0;">model_path: <strong>{{ distillModelPath ?? 'n/a' }}</strong></p>
          </div>
        } @else {
          <p class="muted">Model files were not saved for this run.</p>
        }
        <div class="button-row" style="justify-content: flex-end;">
          <button type="button" class="btn primary" (click)="close()">Close</button>
        </div>
      </div>
    </app-modal>
  `
})
export class DistillMetricsModalComponent {
  @Input() isOpen = false;
  @Input() distillMetrics: any = null;
  @Input() distillModelId: string | null = null;
  @Input() distillModelPath: string | null = null;
  @Input() distillComparison: any = null;
  @Input() close: () => void = () => {};

  readonly formatBytes = formatBytes;
  readonly formatInt = formatInt;
  readonly formatMetricNumber = formatMetricNumber;

  readonly sizeSavedLabel = computed(() =>
    formatPercentLabel({
      value: this.distillComparison?.sizeSavedPercent,
      fallback: '(file-size savings unavailable when no artifact files are persisted)',
    })
  );
  readonly paramsSavedLabel = computed(() => formatPercentLabel({ value: this.distillComparison?.paramSavedPercent, fallback: '' }));
  readonly showModelArtifacts = computed(() =>
    hasModelArtifacts({
      modelId: this.distillModelId ?? undefined,
      modelPath: this.distillModelPath ?? undefined,
    })
  );
}
