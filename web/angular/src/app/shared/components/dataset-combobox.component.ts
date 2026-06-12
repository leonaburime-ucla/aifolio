import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import type { DatasetOption } from '../types/dataset-option';

@Component({
  selector: 'app-dataset-combobox',
  imports: [CommonModule, FormsModule],
  template: `
    <div class="combo">
      <input
        class="combo-input"
        type="text"
        placeholder="Search datasets..."
        [ngModel]="search()"
        (ngModelChange)="onSearchChange($event)"
        (focus)="openList($event)"
        (blur)="closeSoon()"
      />

      @if (isOpen() && filteredOptions().length > 0) {
        <div class="combo-list">
          @for (option of filteredOptions(); track option.id) {
            <div
              class="combo-option"
              [class.active]="option.id === selectedId"
              (mousedown)="selectOption(option)"
            >
              {{ option.label }}
            </div>
          }
        </div>
      }

      @if (isOpen() && filteredOptions().length === 0) {
        <p class="muted" style="font-size: .75rem; margin: .25rem 0 0;">No datasets found.</p>
      }
    </div>
  `
})
export class DatasetComboboxComponent implements OnChanges {
  @Input({ required: true }) options: DatasetOption[] = [];
  @Input() selectedId: string | null = null;
  @Output() changed = new EventEmitter<string>();

  readonly search = signal('');
  readonly isOpen = signal(false);
  readonly isSearching = signal(false);

  readonly filteredOptions = computed(() => {
    const query = this.isOpen() && !this.isSearching() ? '' : this.search().toLowerCase().trim();
    if (!query) return this.options;
    return this.options.filter((option) => option.label.toLowerCase().includes(query));
  });

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['selectedId'] || changes['options']) {
      const match = this.options.find((option) => option.id === this.selectedId);
      if (match && !this.isOpen()) this.search.set(match.label);
    }
  }

  selectOption(option: DatasetOption): void {
    this.search.set(option.label);
    this.isOpen.set(false);
    this.isSearching.set(false);
    this.changed.emit(option.id);
  }

  openList(event: FocusEvent): void {
    this.isOpen.set(true);
    this.isSearching.set(false);
    if (event.target instanceof HTMLInputElement) event.target.select();
  }

  onSearchChange(value: string): void {
    this.search.set(value);
    this.isSearching.set(true);
  }

  closeSoon(): void {
    window.setTimeout(() => {
      this.isOpen.set(false);
      this.isSearching.set(false);
      const match = this.options.find((option) => option.id === this.selectedId);
      if (match) this.search.set(match.label);
    }, 150);
  }
}
