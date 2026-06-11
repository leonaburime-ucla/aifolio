import { CommonModule } from '@angular/common';
import { Component, OnDestroy, signal } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Subscription } from 'rxjs';
import { TrainingScreenComponent } from '../features/ml/components/training-screen.component';
import type { Framework } from '../features/ml/model/ml-training.types';

@Component({
  selector: 'app-training-page',
  imports: [CommonModule, TrainingScreenComponent],
  template: `<app-training-screen [framework]="framework()" />`
})
export class TrainingPageComponent implements OnDestroy {
  readonly framework = signal<Framework>('pytorch');
  private readonly subscription: Subscription;

  constructor(route: ActivatedRoute) {
    this.subscription = route.data.subscribe((data) => {
      this.framework.set((data['framework'] as Framework) ?? 'pytorch');
    });
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }
}
