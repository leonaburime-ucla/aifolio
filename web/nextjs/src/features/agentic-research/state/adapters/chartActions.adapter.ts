import type {
  AgenticResearchChartActionsPort,
  AgenticResearchChartStateSnapshot,
} from "@/features/agentic-research/types/agenticResearch.types";
import { useAgenticResearchChartStore } from "@/features/agentic-research/state/zustand/agenticResearchChartStore";

/**
 * Non-hook function to get a snapshot of the chart state.
 * Used by orchestrators that need synchronous access to current state.
 */
export function getAgenticResearchChartStateSnapshot(): AgenticResearchChartStateSnapshot {
  const { chartSpecs } = useAgenticResearchChartStore.getState();
  return { chartSpecs };
}

/**
 * Adapter hook that exposes Agentic Research chart write actions via a port.
 */
export function useAgenticResearchChartActionsAdapter(): AgenticResearchChartActionsPort {
  const addChartSpec = useAgenticResearchChartStore((state) => state.addChartSpec);
  const clearChartSpecs = useAgenticResearchChartStore((state) => state.clearChartSpecs);
  const removeChartSpec = useAgenticResearchChartStore((state) => state.removeChartSpec);
  const reorderChartSpecs = useAgenticResearchChartStore((state) => state.reorderChartSpecs);

  return {
    addChartSpec,
    clearChartSpecs,
    removeChartSpec,
    reorderChartSpecs,
    getChartStateSnapshot: getAgenticResearchChartStateSnapshot,
  };
}
