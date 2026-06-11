import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, ElementRef, EventEmitter, HostListener, Input, OnChanges, OnDestroy, Output, ViewChild } from '@angular/core';
import * as echarts from 'echarts';
import type { ChartSpec } from '@aifolio/contracts/entities/chart';
import { buildChartOption } from '../lib/build-chart-option';

@Component({
  selector: 'app-chart-renderer',
  imports: [CommonModule],
  template: `
    <div class="chart-card">
      @if (removable) {
        <button type="button" class="remove-fab" aria-label="Remove chart" (click)="remove.emit(spec.id)">x</button>
      }
      <div style="margin-bottom: 1rem;">
        @if (spec.title) {
          <p class="chart-title">{{ spec.title }}</p>
        }
        @if (spec.description) {
          <p class="chart-description">{{ spec.description }}</p>
        }
      </div>
      <div #chartEl class="chart-host"></div>
      @if (spec.meta?.datasetLabel || spec.meta?.queryTimeMs != null) {
        <div class="chart-description">
          @if (spec.meta?.datasetLabel) {
            <p>Dataset: {{ spec.meta?.datasetLabel }}</p>
          }
          @if (spec.meta?.queryTimeMs != null) {
            <p>Query time: {{ ((spec.meta?.queryTimeMs ?? 0) / 1000).toFixed(2) }}s</p>
          }
        </div>
      }
    </div>
  `
})
export class ChartRendererComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input({ required: true }) spec!: ChartSpec;
  @Input() removable = false;
  @Output() remove = new EventEmitter<string>();
  @ViewChild('chartEl') chartEl?: ElementRef<HTMLElement>;

  private chart: echarts.ECharts | null = null;

  ngAfterViewInit(): void {
    if (!this.chartEl) return;
    this.chart = echarts.init(this.chartEl.nativeElement);
    this.render();
  }

  ngOnChanges(): void {
    this.render();
  }

  ngOnDestroy(): void {
    this.chart?.dispose();
    this.chart = null;
  }

  @HostListener('window:resize')
  onResize(): void {
    this.chart?.resize();
  }

  private render(): void {
    if (!this.chart || !this.spec) return;
    this.chart.setOption(buildChartOption(this.spec) as echarts.EChartsOption, true);
  }
}
