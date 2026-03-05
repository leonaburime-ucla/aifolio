"use client";

import type { EChartsRendererProps } from "@/features/recharts/__types__/typescript/react/views/echartsRenderer.types";
import { useEChartsRenderer } from "@/features/recharts/typescript/react/hooks/useEChartsRenderer.hooks";

/**
 * Render charts via ECharts for types not supported by Recharts.
 * Currently supports: heatmap, box, dendrogram.
 * @param spec - ChartSpec configuration.
 */
export default function EChartsRenderer({ spec }: EChartsRendererProps) {
  const { containerRef, option } = useEChartsRenderer({ spec });

  if (!option) return null;

  return <div ref={containerRef} className="h-64 w-full" />;
}
