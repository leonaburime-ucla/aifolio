"use client";

import dynamic from "next/dynamic";
import { useAgenticResearchChatOrchestrator } from "@/screens/AgenticResearchPage/chat/orchestrators/agenticResearchChatOrchestrator";
import type { ChatOrchestrator } from "@/features/ai-chat/typescript/react/orchestrators/chatOrchestrator";
import ChartRenderer from "@/features/recharts/typescript/react/views/components/ChartRenderer";
import DataTable from "@/core/views/components/Datatable/DataTable";
import { useAgenticResearchOrchestrator } from "@/features/agentic-research/typescript/react/orchestrators/agenticResearchOrchestrator";
import DatasetCombobox from "@/features/agentic-research/typescript/react/views/components/DatasetCombobox";
import type { AgenticResearchOrchestratorModel } from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

type AgenticResearchPageProps = {
  pageOrchestrator?: () => AgenticResearchOrchestratorModel;
  chatOrchestrator?: () => ChatOrchestrator;
  showChatSidebar?: boolean;
  algorithmsAccordionInitiallyOpen?: boolean;
};

const ChatSidebar = dynamic(
  () => import("@/features/ai-chat/typescript/react/views/components/ChatSidebar"),
  { ssr: false }
);

export default function AgenticResearchPageScreen({
  pageOrchestrator = useAgenticResearchOrchestrator,
  chatOrchestrator = useAgenticResearchChatOrchestrator,
  showChatSidebar = true,
  algorithmsAccordionInitiallyOpen = true,
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
    removeChartSpec,
    groupedTools,
    formatToolName,
  } = pageOrchestrator();

  const content = (
    <>
      <details
        className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black"
        open={algorithmsAccordionInitiallyOpen}
      >
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

      <details className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-[12px] text-zinc-600">
        <summary className="cursor-pointer font-semibold text-zinc-900">
          Preprocessing Notes
        </summary>
        <div className="mt-3 flex flex-col gap-2">
          <p>
            <strong>Categorical Encoding:</strong> Text columns with &le; 20 unique values are
            automatically One-Hot Encoded.
          </p>
          <p>
            <strong>High Cardinality &amp; IDs:</strong> Text columns with &gt; 20 unique values
            or ID-like names are dropped to prevent feature explosion.
          </p>
          <p>
            <strong>Date Parsing:</strong> Dates and timestamps are extracted into Year, Month,
            and Day numeric features.
          </p>
          <p>
            <strong>Missing Values:</strong> Missing numeric values are imputed using the column
            median to maintain robustness against outliers.
          </p>
          <p>
            <strong>Feature Scaling:</strong> All features are standardized to zero mean and unit
            variance (StandardScaler) before analysis. This prevents large-range features from
            dominating algorithms like PCA.
          </p>
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
                className={`flex flex-col gap-4 ${chartSpecs.length > 2 ? "max-h-[56rem] overflow-y-auto pr-2" : ""
                  }`}
              >
                {chartSpecs.map((spec) => (
                  <ChartRenderer key={spec.id} spec={spec} onRemove={removeChartSpec} />
                ))}
              </div>
            ) : activeChartSpec ? (
              <ChartRenderer spec={activeChartSpec} onRemove={removeChartSpec} />
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
    </>
  );

  /* When embedded (no sidebar), skip the page-level wrappers to avoid double padding */
  if (!showChatSidebar) {
    return <div className="flex flex-col gap-4">{content}</div>;
  }

  return (
    <div className="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
      <main className="min-w-0 flex-1 py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-6 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            Agentic Research
          </p>
          {content}
        </div>
      </main>

      {showChatSidebar ? (
        <div className="sticky top-16 h-[calc(100vh-64px)] w-[360px] shrink-0 overflow-hidden">
          <ChatSidebar
            chatOrchestrator={chatOrchestrator}
            className="!h-full border-l-0"
          />
        </div>
      ) : null}
    </div>
  );
}
