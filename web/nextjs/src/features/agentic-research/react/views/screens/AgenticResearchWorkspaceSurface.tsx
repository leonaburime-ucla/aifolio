"use client";

import type { AgenticResearchOrchestratorModel } from "@aifolio/contracts/entities/agentic-research";
import { useAgenticResearchOrchestrator } from "@/features/agentic-research/react/orchestrators/agenticResearchOrchestrator";
import DatasetCombobox from "@/features/agentic-research/react/views/components/DatasetCombobox";
import ChartRenderer from "@/features/recharts/react/views/components/ChartRenderer";
import DataTable from "@/ui/components/Datatable/DataTable";

const DEFAULT_SAMPLE_PROMPTS = [
  "Run PCA analysis",
  "Run NMF Decomposition and PLSR",
] as const;

type AgenticResearchWorkspaceSurfaceProps = {
  pageOrchestrator?: () => AgenticResearchOrchestratorModel;
  algorithmsAccordionInitiallyOpen?: boolean;
  algorithmsAccordionTitle?: string;
  showAlgorithmsResultsCallout?: boolean;
  showAlgorithmsSamplePrompts?: boolean;
  samplePrompts?: readonly string[];
};

export default function AgenticResearchWorkspaceSurface({
  pageOrchestrator = useAgenticResearchOrchestrator,
  algorithmsAccordionInitiallyOpen = true,
  algorithmsAccordionTitle = "ML Algorithms + Sample Prompts",
  showAlgorithmsResultsCallout = true,
  showAlgorithmsSamplePrompts = true,
  samplePrompts = DEFAULT_SAMPLE_PROMPTS,
}: AgenticResearchWorkspaceSurfaceProps) {
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

  return (
    <>
      <details
        className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black"
        open={algorithmsAccordionInitiallyOpen}
      >
        <summary className="cursor-pointer text-[12px] font-semibold">
          {algorithmsAccordionTitle}
        </summary>
        <div className="mt-3 text-[12px]">
          {showAlgorithmsResultsCallout ? (
            <p className="font-bold text-red-600">Results take 1-2min</p>
          ) : null}
          {showAlgorithmsSamplePrompts ? (
            <div className="mt-3">
              <p className="font-bold text-zinc-900">Sample Prompts</p>
              <ol className="mt-2 list-decimal space-y-1 pl-4">
                {samplePrompts.map((prompt) => (
                  <li key={prompt}>{prompt}</li>
                ))}
              </ol>
            </div>
          ) : null}
          {sklearnTools.length === 0 ? (
            <p className="mt-1">Loading...</p>
          ) : (
            <div
              className={`${
                showAlgorithmsResultsCallout || showAlgorithmsSamplePrompts
                  ? "mt-4"
                  : "mt-1"
              } flex flex-col gap-2`}
            >
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
                    <p className="text-[12px]">
                      {tools.map(formatToolName).join(", ")}
                    </p>
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
            <strong>Categorical Encoding:</strong> Text columns with &le; 20
            unique values are automatically One-Hot Encoded.
          </p>
          <p>
            <strong>High Cardinality &amp; IDs:</strong> Text columns with &gt;
            20 unique values or ID-like names are dropped to prevent feature
            explosion.
          </p>
          <p>
            <strong>Date Parsing:</strong> Dates and timestamps are extracted
            into Year, Month, and Day numeric features.
          </p>
          <p>
            <strong>Missing Values:</strong> Missing numeric values are imputed
            using the column median to maintain robustness against outliers.
          </p>
          <p>
            <strong>Feature Scaling:</strong> All features are standardized to
            zero mean and unit variance (StandardScaler) before analysis. This
            prevents large-range features from dominating algorithms like PCA.
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
        <details
          className="rounded-2xl border border-zinc-200 bg-white/60 p-4 shadow-sm backdrop-blur-sm"
          open
        >
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
                  <ChartRenderer
                    key={spec.id}
                    spec={spec}
                    onRemove={removeChartSpec}
                  />
                ))}
              </div>
            ) : activeChartSpec ? (
              <ChartRenderer
                spec={activeChartSpec}
                onRemove={removeChartSpec}
              />
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
}
