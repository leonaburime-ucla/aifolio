"use client";

import ChatSidebar from "@/features/ai/views/components/ChatSidebar";
import { useAgenticResearchChatOrchestrator } from "@/features/ai/orchestrators/agenticResearchChatOrchestrator";
import type { ChatOrchestrator } from "@/features/ai/orchestrators/chatOrchestrator";
import ChartRenderer from "@/features/recharts/views/components/ChartRenderer";
import DataTable from "@/core/views/components/Datatable/DataTable";
import { useAgenticResearchOrchestrator } from "@/features/agentic-research/orchestrators/agenticResearchOrchestrator";
import DatasetCombobox from "@/features/agentic-research/views/components/DatasetCombobox";
import type { AgenticResearchOrchestratorModel } from "@/features/agentic-research/types/agenticResearch.types";

type AgenticResearchPageProps = {
  pageOrchestrator?: () => AgenticResearchOrchestratorModel;
  chatOrchestrator?: () => ChatOrchestrator;
};

export default function AgenticResearchPage({
  pageOrchestrator = useAgenticResearchOrchestrator,
  chatOrchestrator = useAgenticResearchChatOrchestrator,
}: AgenticResearchPageProps) {
  const {
    isLoading,
    error,
    datasetOptions,
    selectedDatasetId,
    setSelectedDatasetId,
    sklearnTools,
    tableRows,
    tableColumns,
    activeChartSpec,
    chartSpecs,
    groupedTools,
    formatToolName,
  } = pageOrchestrator();

  return (
    <div className="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
      <main className="min-w-0 flex-1 py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-6 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            Agentic Research
          </p>
          <details className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black" open>
            <summary className="cursor-pointer text-[12px] font-semibold">
              Ask Chat to run these Sklearn Algorithms on datasets(Results take 1-2min)
            </summary>
            <div className="mt-3 text-[12px]">
              <p>You can run these algorithms for the current dataset that is in the datatable.</p>
              {sklearnTools.length === 0 ? (
                <p className="mt-1">Loading...</p>
              ) : (
                <div className="mt-2 flex flex-col gap-2">
                  {[
                    "Decomposition & Embeddings",
                    "Classification",
                    "Clustering",
                    "Regression",
                    // "Preprocessing",
                    // "Feature Selection",
                    // "Model Selection",
                    // "Metrics",
                    // "Other",
                  ].map((group) => {
                    const tools = groupedTools[group];
                    if (!tools || tools.length === 0) return null;
                    return (
                      <div key={group}>
                        <p className="text-[11px] font-semibold uppercase tracking-wide">
                          {group}
                        </p>
                        <p className="text-[12px]">{tools.map(formatToolName).join(", ")}</p>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </details>

          <div className="mt-4">
            <div className="mb-4 flex flex-col gap-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                Dataset
              </p>
              <DatasetCombobox
                options={datasetOptions}
                selectedId={selectedDatasetId}
                onChange={setSelectedDatasetId}
              />
            </div>
            <details className="rounded-2xl border border-zinc-200 bg-white/60 p-4 shadow-sm backdrop-blur-sm" open>
              <summary className="cursor-pointer text-sm font-semibold text-zinc-900">
                Charts
              </summary>
              <div className="mt-4">
                {isLoading ? (
                  <div className="h-56 animate-pulse rounded-xl bg-zinc-100" />
                ) : chartSpecs.length > 0 ? (
                  <div
                    className={`flex flex-col gap-4 ${
                      chartSpecs.length > 2 ? "max-h-[56rem] overflow-y-auto pr-2" : ""
                    }`}
                  >
                    {chartSpecs.map((spec) => (
                      <ChartRenderer key={spec.id} spec={spec} />
                    ))}
                  </div>
                ) : activeChartSpec ? (
                  <ChartRenderer spec={activeChartSpec} />
                ) : (
                  <div className="rounded-xl border border-dashed border-zinc-200 bg-zinc-50 px-4 py-6 text-sm text-zinc-500">
                    {error ?? "No analysis chart data available yet."}
                  </div>
                )}
              </div>
            </details>
          </div>

          <DataTable
            key={selectedDatasetId ?? "dataset"}
            rows={tableRows}
            columns={tableColumns}
          />
        </div>
      </main>

      <div className="sticky top-16 h-[calc(100vh-64px)] w-[360px] shrink-0 overflow-hidden">
        <ChatSidebar
          chatOrchestrator={chatOrchestrator}
          className="!h-full border-l-0"
        />
      </div>
    </div>
  );
}
