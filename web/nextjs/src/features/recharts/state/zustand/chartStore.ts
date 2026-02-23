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
      return { chartSpecs: next };
    }),
  removeChartSpec: (id) =>
    set((state) => {
      const next = state.chartSpecs.filter((spec) => spec.id !== id);
      return { chartSpecs: next };
    }),
  clearChartSpecs: () =>
    set(() => ({ chartSpecs: [] })),
}));

export type { ChartStoreState };
