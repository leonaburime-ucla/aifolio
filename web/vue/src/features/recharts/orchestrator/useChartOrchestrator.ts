import { ref, onMounted, watch, onBeforeUnmount, type Ref } from "vue";
import * as echarts from "echarts";
import { storeToRefs } from "pinia";
import { useChartStore } from "~/composables/useChartStore";
import type { ChartSpec } from "~/composables/useChartStore";
import { buildChartOption } from "../lib";

export function useChartRendererOrchestrator(spec: Ref<ChartSpec>) {
  const chartEl = ref<HTMLElement | null>(null);
  let chart: echarts.ECharts | null = null;

  onMounted(() => {
    if (chartEl.value) {
      chart = echarts.init(chartEl.value);
      chart.setOption(buildChartOption(spec.value) as echarts.EChartsOption);
    }
  });

  watch(spec, (newSpec) => {
    chart?.setOption(buildChartOption(newSpec) as echarts.EChartsOption, true);
  }, { deep: true });

  onBeforeUnmount(() => { chart?.dispose(); });

  return { chartEl };
}

export function useChartsWorkspaceOrchestrator() {
  const store = useChartStore();
  const { chartSpecs } = storeToRefs(store);

  return {
    chartSpecs,
    removeChartSpec: store.removeChartSpec,
  };
}
