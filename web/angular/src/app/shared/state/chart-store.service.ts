import { Injectable, computed, signal } from '@angular/core';
import type { ChartSpec } from '@aifolio/contracts/entities/chart';

@Injectable({ providedIn: 'root' })
export class ChartStoreService {
  private readonly specs = signal<ChartSpec[]>([]);
  readonly chartSpecs = computed(() => this.specs());

  addChartSpec(spec: ChartSpec): void {
    this.specs.update((current) => [spec, ...current.filter((item) => item.id !== spec.id)]);
  }

  removeChartSpec(id: string): void {
    this.specs.update((current) => current.filter((spec) => spec.id !== id));
  }

  clearCharts(): void {
    this.specs.set([]);
  }
}
