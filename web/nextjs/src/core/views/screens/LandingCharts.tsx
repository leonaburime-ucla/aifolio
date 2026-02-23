"use client";

import ChartRenderer from "@/features/recharts/views/components/ChartRenderer";
import { useChartOrchestrator } from "@/features/recharts/orchestrators/chartOrchestrator";

type LandingChartsProps = {
  orchestrator?: typeof useChartOrchestrator;
};

export default function LandingCharts({
  orchestrator = useChartOrchestrator,
}: LandingChartsProps) {
  const { chartSpecs, removeChartSpec } = orchestrator();

  return (
    <div className="flex flex-col gap-8">
      {/* <div className="rounded-xl border border-zinc-200 bg-zinc-100 px-3 py-2 text-xs text-zinc-700">
        {`Chart Count: ${chartSpecs.length} | ids: ${
          chartSpecs.map((spec) => spec.id).join(", ") || "none"
        }`}
      </div> */}
      {chartSpecs.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-zinc-300 bg-white p-6 text-sm text-zinc-500">
          Charts generated from chat will appear here.
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          {chartSpecs.map((spec) => (
            <div
              key={spec.id}
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
          ))}
        </div>
      )}
    </div>
  );
}
