"use client";

import { useMemo } from "react";
import BitcoinCharts from "@/features/recharts/views/components/BitcoinCharts";
import ChartRenderer from "@/features/recharts/views/components/ChartRenderer";
import { useChartOrchestrator } from "@/features/recharts/orchestrators/chartOrchestrator";

type LandingChartsProps = {
  coin: any;
  chart: any;
};

export default function LandingCharts({ coin, chart }: LandingChartsProps) {
  const { chartSpecs, removeChartSpec } = useChartOrchestrator();

  const chartNodes = useMemo(
    () =>
      chartSpecs.map((spec, index) => (
        <div
          key={`${spec.id}-${index}`}
          className="relative"
        >
          <button
            type="button"
            onClick={() => removeChartSpec(spec.id)}
            aria-label="Remove chart"
            className="absolute -right-2 -top-2 z-10 flex h-7 w-7 items-center justify-center rounded-full border border-zinc-200 bg-white text-zinc-500 shadow-sm transition hover:bg-zinc-50"
          >
            ×
          </button>
          <ChartRenderer spec={spec} />
        </div>
      )),
    [chartSpecs, removeChartSpec]
  );

  return (
    <div className="flex flex-col gap-8">
      {chartNodes.length > 0 ? (
        <div className="flex flex-col gap-6">{chartNodes}</div>
      ) : null}

      <div className="flex flex-col gap-6">
        <BitcoinCharts coin={coin} chart={chart} />
        <BitcoinCharts coin={coin} chart={chart} />
      </div>
    </div>
  );
}
