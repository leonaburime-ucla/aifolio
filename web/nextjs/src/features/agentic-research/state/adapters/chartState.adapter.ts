import type { AgenticResearchChartStatePort } from "@/features/agentic-research/types/agenticResearch.types";
import { useAgenticResearchChartStore } from "@/features/agentic-research/state/zustand/agenticResearchChartStore";

/**
 * Adapter hook that exposes chart state through the Agentic Research chart-state port.
 * @returns Chart state port model for orchestrators.
 */
export function useAgenticResearchChartStateAdapter(): AgenticResearchChartStatePort {
  const chartSpecs = useAgenticResearchChartStore((state) => state.chartSpecs);
  const removeChartSpec = useAgenticResearchChartStore((state) => state.removeChartSpec);
  return { chartSpecs, removeChartSpec };
}
