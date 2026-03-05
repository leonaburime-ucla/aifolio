import { create } from "zustand";
import {
  addChartSpecDedupPrepend,
  reorderChartSpecsWithRemainder,
} from "@/features/agentic-research/typescript/logic/agenticResearchChartStore.logic";
import type { AgenticResearchChartStoreState } from "@/features/agentic-research/__types__/typescript/react/state/agenticResearchChartStore.types";

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
