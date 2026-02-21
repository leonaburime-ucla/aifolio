import { useChartStore } from "@/features/recharts/state/zustand/chartStore";
import type { ChartSpec } from "@/features/ai/types/chart.types";

export type ChartStatePort = {
  chartSpecs: ChartSpec[];
};

/**
 * Adapter hook that exposes chart state through a neutral chart-state port.
 * @returns Chart state port model for orchestrators/hooks.
 */
export function useChartStateAdapter(): ChartStatePort {
  const chartSpecs = useChartStore((state) => state.chartSpecs);
  return { chartSpecs };
}
