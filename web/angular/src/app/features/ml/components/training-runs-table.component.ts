import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output, computed, signal } from '@angular/core';
import { TRAINING_RUN_COLUMNS, buildDistillActionModel, calcTrainingTableHeight } from '@aifolio/frontend-core/ml-training';
import type { TrainingRunRow } from '@aifolio/contracts/entities/ml-training';
import type { Framework } from '../model/ml-training.types';

@Component({
  selector: 'app-training-runs-table',
  imports: [CommonModule],
  template: `
    <div class="training-table-wrap" [style.height.px]="tableHeight()">
      <table class="training-table">
        <thead>
          <tr>
            @for (column of columns; track column) {
              <th>
                <button type="button" class="btn secondary" (click)="toggleSort(column)">
                  {{ column }}
                  @if (sortKey() === column) {
                    <span>{{ sortDirection() === 'asc' ? '▲' : '▼' }}</span>
                  }
                </button>
              </th>
            }
          </tr>
        </thead>
        <tbody>
          @for (row of sortedRows(); track rowKey(row, $index)) {
            <tr>
              @for (column of columns; track column) {
                <td>
                  @if (column === 'distill_action') {
                    @let action = actionModel(row);
                    @if (action.kind === 'student_model') {
                      <span class="status-pill">Student Model</span>
                    } @else if (action.kind === 'not_available') {
                      <span class="muted">Not Available</span>
                    } @else if (action.kind === 'show_distilled') {
                      <button type="button" class="btn secondary" (click)="seeDistilled.emit(row)">Show Distilled</button>
                    } @else {
                      <button type="button" class="btn primary" [disabled]="action.isDistillingThisRow" (click)="distill.emit(row)">
                        {{ action.isDistillingThisRow ? 'Distilling...' : 'Distill' }}
                      </button>
                    }
                  } @else {
                    {{ formatCell(row[column]) }}
                  }
                </td>
              }
            </tr>
          }
        </tbody>
      </table>
    </div>
  `
})
export class TrainingRunsTableComponent {
  @Input() runs: TrainingRunRow[] = [];
  @Input() framework: Framework = 'pytorch';
  @Input() distillingTeacherKey: string | null = null;
  @Input() distilledByTeacher: Record<string, string> = {};
  @Output() distill = new EventEmitter<TrainingRunRow>();
  @Output() seeDistilled = new EventEmitter<TrainingRunRow>();

  readonly columns = [...TRAINING_RUN_COLUMNS];
  readonly sortKey = signal<string | null>(null);
  readonly sortDirection = signal<'asc' | 'desc'>('asc');
  readonly tableHeight = computed(() => calcTrainingTableHeight({ rowsCount: this.runs.length }));

  readonly sortedRows = computed(() => {
    if (!this.sortKey()) return this.runs;
    const direction = this.sortDirection() === 'asc' ? 1 : -1;
    return [...this.runs].sort((left, right) => this.compare(left[this.sortKey()!], right[this.sortKey()!]) * direction);
  });

  toggleSort(column: string): void {
    if (this.sortKey() === column) {
      this.sortDirection.set(this.sortDirection() === 'asc' ? 'desc' : 'asc');
      return;
    }
    this.sortKey.set(column);
    this.sortDirection.set('asc');
  }

  actionModel(row: TrainingRunRow) {
    return buildDistillActionModel({
      row,
      isDistillationSupportedForRun: (candidate) => this.isDistillationSupportedForRun(candidate),
      distillingTeacherKey: this.distillingTeacherKey,
      distilledByTeacher: this.distilledByTeacher,
    });
  }

  isDistillationSupportedForRun(row: Partial<TrainingRunRow>): boolean {
    const mode = String(row.training_mode ?? '');
    if (this.framework === 'tensorflow') return ['mlp_dense', 'linear_glm_baseline', 'wide_and_deep'].includes(mode);
    return ['mlp_dense', 'linear_glm_baseline', 'tabresnet'].includes(mode);
  }

  compare(left: unknown, right: unknown): number {
    const leftNumber = Number(left);
    const rightNumber = Number(right);
    if (Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber - rightNumber;
    return String(left ?? '').localeCompare(String(right ?? ''), undefined, { numeric: true, sensitivity: 'base' });
  }

  formatCell(value: unknown): string {
    return value == null ? '' : String(value);
  }

  rowKey(row: TrainingRunRow, index: number): string {
    return String(row.run_id ?? row.model_id ?? row.completed_at ?? index);
  }
}
