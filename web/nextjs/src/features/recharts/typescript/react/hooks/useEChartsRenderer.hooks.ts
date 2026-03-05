"use client";

import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";
import { getEChartsOption } from "@/features/recharts/typescript/logic/echartsOptions.logic";
import type {
  EChartsRendererRuntime,
  UseEChartsRendererParams,
  UseEChartsRendererResult,
} from "@/features/recharts/__types__/typescript/react/hooks/echartsRenderer.hooks.types";

const DEFAULT_ECHARTS_RUNTIME: EChartsRendererRuntime = {
  initChart: (container) => echarts.init(container),
  bindResize: (onResize) => {
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
    };
  },
};

/**
 * Handles ECharts option memoization plus mount/unmount lifecycle binding.
 * @param params - Required parameters.
 * @param params.spec - Chart specification consumed by ECharts.
 * @returns Container ref and computed option.
 */
export function useEChartsRenderer({
  spec,
}: UseEChartsRendererParams, {
  runtime = DEFAULT_ECHARTS_RUNTIME,
}: { runtime?: EChartsRendererRuntime } = {}): UseEChartsRendererResult {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const option = useMemo(() => getEChartsOption({ spec }), [spec]);

  useEffect(() => {
    if (!containerRef.current || !option) return;
    const chart = runtime.initChart(containerRef.current);
    chart.setOption(option);

    const resize = () => chart.resize();
    const unbindResize = runtime.bindResize(resize);
    return () => {
      unbindResize();
      chart.dispose();
    };
  }, [option, runtime]);

  return { containerRef, option };
}
