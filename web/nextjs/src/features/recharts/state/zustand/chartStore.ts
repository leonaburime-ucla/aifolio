import { create } from "zustand";
import type { ChartSpec } from "@/features/ai/types/chart.types";

type ChartStoreState = {
  chartSpecs: ChartSpec[];
  addChartSpec: (spec: ChartSpec) => void;
  removeChartSpec: (id: string) => void;
  clearChartSpecs: () => void;
};

export const useChartStore = create<ChartStoreState>((set) => ({
  chartSpecs: [],
  addChartSpec: (spec) =>
    set((state) => {
      const next = [spec, ...state.chartSpecs.filter((item) => item.id !== spec.id)];
      console.log("[chart-store] addChartSpec", {
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
      console.log("[chart-store] removeChartSpec", {
        removedId: id,
        nextCount: next.length,
        nextIds: next.map((item) => item.id),
      });
      return { chartSpecs: next };
    }),
  clearChartSpecs: () =>
    set((state) => {
      console.log("[chart-store] clearChartSpecs", {
        previousCount: state.chartSpecs.length,
        previousIds: state.chartSpecs.map((item) => item.id),
      });
      return { chartSpecs: [] };
    }),
}));

export type { ChartStoreState };
