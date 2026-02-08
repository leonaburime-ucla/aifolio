import { create } from "zustand";
import type { ChartSpec } from "@/features/ai/types/chart.types";

type ChartStoreState = {
  chartSpecs: ChartSpec[];
  addChartSpec: (spec: ChartSpec) => void;
  removeChartSpec: (id: string) => void;
};

export const useChartStore = create<ChartStoreState>((set) => ({
  chartSpecs: [],
  addChartSpec: (spec) =>
    set((state) => ({ chartSpecs: [spec, ...state.chartSpecs] })),
  removeChartSpec: (id) =>
    set((state) => ({
      chartSpecs: state.chartSpecs.filter((spec) => spec.id !== id),
    })),
}));

export type { ChartStoreState };
