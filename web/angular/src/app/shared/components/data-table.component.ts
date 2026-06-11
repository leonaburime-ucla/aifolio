import { CommonModule } from '@angular/common';
import { Component, Input, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-data-table',
  imports: [CommonModule, FormsModule],
  template: `
    @if (columns.length > 0) {
      <div class="data-table-wrap">
        <div class="table-toolbar">
          <span class="muted">{{ rows.length }} rows</span>
          <input
            type="search"
            placeholder="Search table..."
            [ngModel]="search()"
            (ngModelChange)="search.set($event); page.set(0)"
          />
        </div>
        <table class="data-table">
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
            @for (row of pagedRows(); track rowIndex($index)) {
              <tr>
                @for (column of columns; track column) {
                  <td>{{ formatCell(row[column]) }}</td>
                }
              </tr>
            }
          </tbody>
        </table>
        <div class="table-toolbar">
          <button type="button" class="btn secondary" [disabled]="page() === 0" (click)="page.set(page() - 1)">Previous</button>
          <span class="muted">Page {{ page() + 1 }} of {{ pageCount() }}</span>
          <button type="button" class="btn secondary" [disabled]="page() + 1 >= pageCount()" (click)="page.set(page() + 1)">Next</button>
        </div>
      </div>
    }
  `
})
export class DataTableComponent {
  @Input() rows: Record<string, unknown>[] = [];
  @Input() columns: string[] = [];

  readonly search = signal('');
  readonly sortKey = signal<string | null>(null);
  readonly sortDirection = signal<'asc' | 'desc'>('asc');
  readonly page = signal(0);
  readonly pageSize = 25;

  readonly filteredRows = computed(() => {
    const query = this.search().toLowerCase().trim();
    if (!query) return this.rows;
    return this.rows.filter((row) =>
      this.columns.some((column) => String(row[column] ?? '').toLowerCase().includes(query))
    );
  });

  readonly sortedRows = computed(() => {
    const key = this.sortKey();
    if (!key) return this.filteredRows();
    const direction = this.sortDirection() === 'asc' ? 1 : -1;
    return [...this.filteredRows()].sort((left, right) => this.compare(left[key], right[key]) * direction);
  });

  readonly pageCount = computed(() => Math.max(1, Math.ceil(this.sortedRows().length / this.pageSize)));

  readonly pagedRows = computed(() => {
    const start = Math.min(this.page(), this.pageCount() - 1) * this.pageSize;
    return this.sortedRows().slice(start, start + this.pageSize);
  });

  toggleSort(column: string): void {
    if (this.sortKey() === column) {
      this.sortDirection.set(this.sortDirection() === 'asc' ? 'desc' : 'asc');
      return;
    }
    this.sortKey.set(column);
    this.sortDirection.set('asc');
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

  rowIndex(index: number): number {
    return index;
  }
}
