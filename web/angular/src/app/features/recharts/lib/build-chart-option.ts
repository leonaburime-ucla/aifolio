import { getEChartsOption } from '@aifolio/frontend-core/recharts';
import type { ChartSpec } from '@aifolio/contracts/entities/chart';

export type EChartsOption = Record<string, unknown>;

export function buildChartOption(spec: ChartSpec): EChartsOption {
  if (!spec.data || spec.data.length === 0) {
    return { tooltip: {}, xAxis: { type: 'category', data: [] }, yAxis: { type: 'value' }, series: [] };
  }

  const shared = getEChartsOption({ spec });
  if (shared) {
    if (spec.type === 'scatter' || spec.type === 'biplot') {
      const yKey = spec.yKeys[0];
      return {
        ...shared,
        dataset: { source: spec.data },
        series: [
          {
            ...((shared as { series?: Array<Record<string, unknown>> }).series?.[0] ?? {}),
            data: undefined,
            encode: { x: spec.xKey, y: yKey },
          },
        ],
      };
    }
    return shared as EChartsOption;
  }

  const xKey = spec.xKey || Object.keys(spec.data[0])[0] || 'x';
  const xData = spec.data.map((row) => String(row[xKey] ?? ''));

  if (spec.type === 'line' || spec.type === 'area') {
    const series = spec.yKeys.map((key) => ({
      name: key,
      type: 'line',
      data: spec.data.map((row) => Number(row[key] ?? 0)),
      ...(spec.type === 'area' ? { areaStyle: {} } : {}),
    }));
    return {
      tooltip: { trigger: 'axis' },
      legend: spec.yKeys.length > 1 ? { data: spec.yKeys } : undefined,
      xAxis: { type: 'category', data: xData, name: spec.xLabel },
      yAxis: { type: 'value', name: spec.yLabel },
      series,
    };
  }

  const yKey = spec.yKeys[0] || 'y';
  return {
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: xData, name: spec.xLabel },
    yAxis: { type: 'value', name: spec.yLabel },
    series: [{ name: yKey, type: spec.type, data: spec.data.map((row) => Number(row[yKey] ?? 0)) }],
  };
}
