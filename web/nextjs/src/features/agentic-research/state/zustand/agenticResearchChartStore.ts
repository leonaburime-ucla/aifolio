import { create } from "zustand";
import type { ChartSpec } from "@/features/ai/types/chart.types";

type AgenticResearchChartStoreState = {
  chartSpecs: ChartSpec[];
  addChartSpec: (spec: ChartSpec) => void;
  removeChartSpec: (id: string) => void;
  clearChartSpecs: () => void;
  reorderChartSpecs: (orderedIds: string[]) => void;
};

export const useAgenticResearchChartStore = create<AgenticResearchChartStoreState>((set) => ({
  chartSpecs: [],
  addChartSpec: (spec) =>
    set((state) => {
      const next = [spec, ...state.chartSpecs.filter((item) => item.id !== spec.id)];
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
      const byId = new Map(state.chartSpecs.map((spec) => [spec.id, spec]));
      const reordered = orderedIds
        .map((id) => byId.get(id))
        .filter((spec): spec is ChartSpec => Boolean(spec));
      const remaining = state.chartSpecs.filter((spec) => !orderedIds.includes(spec.id));
      const next = [...reordered, ...remaining];
      return { chartSpecs: next };
    }),
}));

export type { AgenticResearchChartStoreState };
