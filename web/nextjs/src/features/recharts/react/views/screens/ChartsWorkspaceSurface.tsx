"use client";

import ChartRenderer from "@/features/recharts/react/views/components/ChartRenderer";
import { useChartOrchestrator } from "@/features/recharts/react/orchestrators/chartOrchestrator";

type ChartsWorkspaceSurfaceProps = {
  showUsageGuide?: boolean;
  orchestrator?: typeof useChartOrchestrator;
};

export default function ChartsWorkspaceSurface({
  showUsageGuide = true,
  orchestrator = useChartOrchestrator,
}: ChartsWorkspaceSurfaceProps) {
  const { chartSpecs, removeChartSpec } = orchestrator();

  return (
    <>
      {showUsageGuide ? (
        <details
          className="rounded-2xl border border-zinc-200 bg-white/70 p-4 shadow-sm backdrop-blur-sm"
          open
        >
          <summary className="cursor-pointer text-sm font-semibold text-zinc-900">
            How to Use Page + Prompts to Try
          </summary>
          <div className="mt-3 space-y-3 text-sm text-zinc-700">
            <p>
              This AI Chat creates charts(Recharts and Echarts) from internal
              sample data. I do not have APIs for real-time data. All data is
              from the LLM&apos;s internal models.
            </p>
            <div>
              <p className="font-medium text-zinc-900">Sample prompts to try:</p>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                <li>“Create a line chart of Solana and Bitcoin for the past 5 months.”</li>
                <li>“Create an area chart of Peruvian beef exports over the past 15 years.”</li>
                <li>“Show a line chart of Manhattan vs London vs Paris average rent since 2000 as a share of average salary in each of those cities respectively.”</li>
                <li>“Plot a line chart of US debt levels for the past 50 years. Estimate what it will be for the next 20 in a blue line”</li>
                <li>“Make a scatter chart comparing Bitcoin and Ethereum returns over the last 30 days.”</li>
              </ul>
            </div>
          </div>
        </details>
      ) : null}
      <div className="flex flex-col gap-8">
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
    </>
  );
}
