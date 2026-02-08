"use client";

import ChatSidebar from "@/features/ai/views/components/ChatSidebar";
import ChartRenderer from "@/features/recharts/views/components/ChartRenderer";
import DataTable from "@/components/DataTable";
import { useAgenticResearchOrchestrator } from "@/features/agentic-research/orchestrators/agenticResearchOrchestrator";
import DatasetCombobox from "@/features/agentic-research/views/components/DatasetCombobox";
import { useChartStore } from "@/features/recharts/state/zustand/chartStore";

const ACRONYMS: Record<string, string> = {
  pca: "PCA",
  svd: "SVD",
  ica: "ICA",
  nmf: "NMF",
  tsne: "t-SNE",
  knn: "KNN",
  rfe: "RFE",
  rfecv: "RFECV",
  svr: "SVR",
  svc: "SVC",
  lda: "LDA",
  qda: "QDA",
  gmm: "GMM",
  kmeans: "K-Means",
  minibatch: "Mini-Batch",
  dbscan: "DBSCAN",
  optics: "OPTICS",
  pls: "PLS",
  elasticnet: "ElasticNet",
  minmax: "MinMax",
};

function formatToolName(name: string): string {
  return name
    .split("_")
    .map((word) => ACRONYMS[word] ?? word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export default function AgenticResearchPage({
  pageOrchestrator = useAgenticResearchOrchestrator
}) {
  const {
    pcaChartSpec,
    isLoading,
    error,
    datasetOptions,
    selectedDatasetId,
    setSelectedDatasetId,
    sklearnTools,
    tableRows,
    tableColumns,
  } = pageOrchestrator();
  const chartSpecs = useChartStore((state) => state.chartSpecs);
  const activeChartSpec = pcaChartSpec ?? chartSpecs[0] ?? null;
  const groupedTools = sklearnTools.reduce<Record<string, string[]>>(
    (acc, tool) => {
      let group = "Other";
      if (tool.includes("regression")) group = "Regression";
      else if (tool.includes("classification")) group = "Classification";
      else if (tool.includes("clustering")) group = "Clustering";
      else if (
        tool.includes("pca") ||
        tool.includes("svd") ||
        tool.includes("ica") ||
        tool.includes("nmf") ||
        tool.includes("tsne")
      )
        group = "Decomposition & Embeddings";
      else if (
        tool.includes("scaler") ||
        tool.includes("encoder") ||
        tool.includes("transformer") ||
        tool.includes("imputer")
      )
        group = "Preprocessing";
      else if (
        tool.includes("select_") ||
        tool.includes("rfe") ||
        tool.includes("rfecv")
      )
        group = "Feature Selection";
      else if (tool.includes("train_test_split")) group = "Model Selection";
      else if (
        tool.includes("accuracy") ||
        tool.includes("precision") ||
        tool.includes("recall") ||
        tool.includes("f1") ||
        tool.includes("roc") ||
        tool.includes("auc")
      )
        group = "Metrics";
      if (!acc[group]) acc[group] = [];
      acc[group].push(tool);
      return acc;
    },
    {}
  );

  return (
    <div className="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
      <main className="min-w-0 flex-1 py-10">
        <div className="mx-auto flex max-w-5xl flex-col gap-6 px-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
            Agentic Research
          </p>
          <details className="rounded-lg border border-zinc-200 bg-white px-4 py-3 text-black">
            <summary className="cursor-pointer text-[12px] font-semibold">
              Ask Chat to run these Sklearn Algorithms on datasets
            </summary>
            <div className="mt-3 text-[12px]">
              <p>You can run these algorithms for the datasets below.</p>
              {sklearnTools.length === 0 ? (
                <p className="mt-1">Loading...</p>
              ) : (
                <div className="mt-2 flex flex-col gap-2">
                  {[
                    "Decomposition & Embeddings",
                    "Classification",
                    "Clustering",
                    "Regression",
                    "Preprocessing",
                    "Feature Selection",
                    "Model Selection",
                    "Metrics",
                    "Other",
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
            {isLoading ? (
              <div className="h-56 animate-pulse rounded-xl bg-zinc-100" />
            ) : chartSpecs.length > 0 ? (
              <div className="flex flex-col gap-4">
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

          <DataTable
            key={selectedDatasetId ?? "dataset"}
            rows={tableRows}
            columns={tableColumns}
          />
        </div>
      </main>

      <div className="sticky top-0 h-screen w-[360px] shrink-0 overflow-hidden">
        <ChatSidebar />
      </div>
    </div>
  );
}
