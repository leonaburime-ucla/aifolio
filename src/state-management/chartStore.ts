import { create } from "zustand";

interface ChartState {
  selectedDataset: string;
  setDataset: (dataset: string) => void;
}

export const createChartStore = () =>
  create<ChartState>((set) => ({
    selectedDataset: "",
    setDataset: (dataset) => set({ selectedDataset: dataset }),
  }));
