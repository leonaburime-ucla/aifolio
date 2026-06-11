import { CommonModule } from '@angular/common';
import { Component, EventEmitter, HostListener, Input, OnChanges, OnDestroy, Output } from '@angular/core';

@Component({
  selector: 'app-modal',
  imports: [CommonModule],
  template: `
    @if (isOpen) {
      <div class="modal-backdrop" [class.top]="position === 'top'">
        <div class="modal-scrim" aria-hidden="true" (click)="close.emit()"></div>
        <div class="modal-panel" role="dialog" aria-modal="true">
          <div class="modal-header">
            <h2>{{ title }}</h2>
            <button type="button" class="btn secondary" aria-label="Close modal" (click)="close.emit()">x</button>
          </div>
          <div class="modal-body">
            <ng-content />
          </div>
        </div>
      </div>
    }
  `
})
export class ModalComponent implements OnChanges, OnDestroy {
  @Input() isOpen = false;
  @Input() title = '';
  @Input() position: 'center' | 'top' = 'center';
  @Output() close = new EventEmitter<void>();

  ngOnChanges(): void {
    document.body.style.overflow = this.isOpen ? 'hidden' : '';
  }

  ngOnDestroy(): void {
    document.body.style.overflow = '';
  }

  @HostListener('window:keydown.escape')
  onEscape(): void {
    if (this.isOpen) this.close.emit();
  }
}
