import { create } from "zustand";
import {
  addChartSpecDedupPrepend,
  reorderChartSpecsWithRemainder,
} from "@aifolio/frontend-core/agentic-research";
import type { AgenticResearchChartStoreState } from "@/features/agentic-research/react/state/zustand/agenticResearchChartStore.types";

export const useAgenticResearchChartStore = create<AgenticResearchChartStoreState>((set) => ({
  chartSpecs: [],
  addChartSpec: (spec) =>
    set((state) => {
      const next = addChartSpecDedupPrepend({
        chartSpecs: state.chartSpecs,
        spec,
      });
      return { chartSpecs: next };
    }),
  removeChartSpec: (id) =>
    set((state) => {
      const next = state.chartSpecs.filter((spec) => spec.id !== id);
      return { chartSpecs: next };
    }),
  clearChartSpecs: () =>
    set(() => ({ chartSpecs: [] })),
  reorderChartSpecs: (orderedIds) =>
    set((state) => {
      const next = reorderChartSpecsWithRemainder({
        chartSpecs: state.chartSpecs,
        orderedIds,
      });
      return { chartSpecs: next };
    }),
}));
