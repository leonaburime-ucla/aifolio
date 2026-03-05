import type { EChartsOption } from "echarts";
import type { MutableRefObject } from "react";
import type { ChartSpec } from "@/features/recharts/__types__/typescript/chart.types";

export type UseEChartsRendererParams = {
  spec: ChartSpec;
};

export type UseEChartsRendererResult = {
  containerRef: MutableRefObject<HTMLDivElement | null>;
  option: EChartsOption | null;
};

export type EChartsInstanceAdapter = {
  setOption: (option: EChartsOption) => void;
  resize: () => void;
  dispose: () => void;
};

export type EChartsRendererRuntime = {
  initChart: (container: HTMLDivElement) => EChartsInstanceAdapter;
  bindResize: (onResize: () => void) => () => void;
};
