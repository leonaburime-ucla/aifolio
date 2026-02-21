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
      console.log("[agentic-research-chart-store] addChartSpec", {
        addedId: spec.id,
        addedType: spec.type,
        nextCount: next.length,
        nextIds: next.map((item) => item.id),
      });
      return { chartSpecs: next };
    }),
  removeChartSpec: (id) =>
    set((state) => {
      const next = state.chartSpecs.filter((spec) => spec.id !== id);
      console.log("[agentic-research-chart-store] removeChartSpec", {
        removedId: id,
        nextCount: next.length,
      });
      return { chartSpecs: next };
    }),
  clearChartSpecs: () =>
    set((state) => {
      console.log("[agentic-research-chart-store] clearChartSpecs", {
        previousCount: state.chartSpecs.length,
      });
      return { chartSpecs: [] };
    }),
  reorderChartSpecs: (orderedIds) =>
    set((state) => {
      const byId = new Map(state.chartSpecs.map((spec) => [spec.id, spec]));
      const reordered = orderedIds
        .map((id) => byId.get(id))
        .filter((spec): spec is ChartSpec => Boolean(spec));
      const remaining = state.chartSpecs.filter((spec) => !orderedIds.includes(spec.id));
      const next = [...reordered, ...remaining];
      console.log("[agentic-research-chart-store] reorderChartSpecs", {
        orderedIds,
        nextIds: next.map((spec) => spec.id),
      });
      return { chartSpecs: next };
    }),
}));

export type { AgenticResearchChartStoreState };
