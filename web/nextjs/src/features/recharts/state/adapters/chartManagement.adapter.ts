import { useChartStore } from "@/features/recharts/state/zustand/chartStore";
import type { ChartSpec } from "@/features/ai/types/chart.types";

export type ChartManagementPort = {
  chartSpecs: ChartSpec[];
  removeChartSpec: (id: string) => void;
};

/**
 * Adapter that exposes chart read/write operations required by chart UIs.
 */
export function useChartManagementAdapter(): ChartManagementPort {
  const chartSpecs = useChartStore((state) => state.chartSpecs);
  const removeChartSpec = useChartStore((state) => state.removeChartSpec);
  return { chartSpecs, removeChartSpec };
}
