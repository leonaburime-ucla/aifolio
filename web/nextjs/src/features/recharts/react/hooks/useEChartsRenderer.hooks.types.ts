import type { MutableRefObject } from "react";
import type { ChartSpec } from "@aifolio/contracts/entities/chart";

export type UseEChartsRendererParams = {
  spec: ChartSpec;
};

export type UseEChartsRendererResult = {
  containerRef: MutableRefObject<HTMLDivElement | null>;
  option: unknown | null;
};

export type EChartsInstanceAdapter = {
  setOption: (option: unknown) => void;
  resize: () => void;
  dispose: () => void;
};

export type EChartsRendererRuntime = {
  initChart: (container: HTMLDivElement) => EChartsInstanceAdapter;
  bindResize: (onResize: () => void) => () => void;
};
