import { useChartStore } from "@/features/recharts/react/state/zustand/chartStore";
import type { ChartManagementPort } from "@aifolio/contracts/entities/chart";

/**
 * Adapter that exposes chart read/write operations required by chart UIs.
 */
export function useChartManagementAdapter(): ChartManagementPort {
  const chartSpecs = useChartStore((state) => state.chartSpecs);
  const removeChartSpec = useChartStore((state) => state.removeChartSpec);
  const clearChartSpecs = useChartStore((state) => state.clearChartSpecs);
  return { chartSpecs, removeChartSpec, clearChartSpecs };
}
