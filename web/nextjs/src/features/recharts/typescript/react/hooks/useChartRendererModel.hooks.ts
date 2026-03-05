"use client";

import { useMemo, useState } from "react";
import { coerceNumber } from "@/features/recharts/typescript/logic/chartFormatting.logic";
import type {
  UseChartRendererModelParams,
  UseChartRendererModelResult,
} from "@/features/recharts/__types__/typescript/react/hooks/chartRenderer.hooks.types";

/**
 * Builds local view-model state for chart rendering components.
 * @param params - Required parameters.
 * @param params.spec - Chart specification rendered by the component.
 * @returns Derived chart props and local expansion state.
 */
export function useChartRendererModel({
  spec,
}: UseChartRendererModelParams): UseChartRendererModelResult {
  const [isExpanded, setIsExpanded] = useState(false);

  const yKeys = useMemo(() => (Array.isArray(spec.yKeys) ? spec.yKeys : []), [spec.yKeys]);

  const scatterLabelKey = useMemo(() => {
    const firstRow = spec.data?.[0] as Record<string, unknown> | undefined;
    if (!firstRow || typeof firstRow !== "object") return null;
    if ("feature" in firstRow) return "feature";
    if ("label" in firstRow) return "label";
    if ("name" in firstRow) return "name";
    return null;
  }, [spec.data]);

  const data = useMemo(() => {
    return (spec.data ?? []).map((row) => {
      const next: Record<string, unknown> = { ...row };
      yKeys.forEach((key) => {
        next[key] = coerceNumber({ value: row[key] });
      });
      return next;
    });
  }, [spec.data, yKeys]);

  const chartProps = useMemo(
    () => ({
      data,
      margin: { top: 12, right: 20, left: 28, bottom: 40 },
    }),
    [data]
  );

  return {
    isExpanded,
    setIsExpanded,
    yKeys,
    scatterLabelKey,
    data,
    chartProps,
  };
}
