import { defineStore } from "pinia";
import { ref } from "vue";
import type { ChartSpec as ContractChartSpec } from "@aifolio/contracts/entities/chart";

export type ChartSpec = ContractChartSpec;

export const useChartStore = defineStore("charts", () => {
  const chartSpecs = ref<ChartSpec[]>([]);

  function addChartSpec(spec: ChartSpec) {
    chartSpecs.value.unshift(spec);
  }

  function removeChartSpec(id: string) {
    chartSpecs.value = chartSpecs.value.filter((s) => s.id !== id);
  }

  function clearCharts() {
    chartSpecs.value = [];
  }

  return { chartSpecs, addChartSpec, removeChartSpec, clearCharts };
});
