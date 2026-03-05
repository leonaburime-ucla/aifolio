import { create } from "zustand";
import type { ChartStoreState } from "@/features/recharts/__types__/typescript/react/state/chartStore.types";

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
