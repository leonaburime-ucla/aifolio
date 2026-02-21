import { fetchPcaChartSpec } from "@/features/agentic-research/api/agenticResearchApi";
import { useAgenticResearchIntegration } from "@/features/agentic-research/hooks/useAgenticResearch.hooks";
import { useChartStateAdapter } from "@/features/agentic-research/state/adapters/chartState.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/state/adapters/agenticResearchState.adapter";
import type {
  AgenticResearchApiDeps,
  AgenticResearchDeps,
  AgenticResearchOrchestratorModel,
  UseAgenticResearchChartStatePort,
  UseAgenticResearchStatePort,
} from "@/features/agentic-research/types/agenticResearch.types";
import { useMemo } from "react";

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

/**
 * Orchestrator hook that wires state + API dependencies into the agentic research hooks.
 * @param deps - Optional adapter overrides for orchestrator ports.
 * @returns UI-ready agentic research model for the page.
 */
export function useAgenticResearchOrchestrator(
  options: {
    useStatePort?: UseAgenticResearchStatePort;
    useChartStatePort?: UseAgenticResearchChartStatePort;
  } = {}
): AgenticResearchOrchestratorModel {
  const {
    useStatePort = useAgenticResearchStateAdapter,
    useChartStatePort = useChartStateAdapter,
  } = options;
  const { state, actions } = useStatePort();
  const { chartSpecs } = useChartStatePort();

  const api = useMemo<AgenticResearchApiDeps>(
    () => ({
      fetchPcaChartSpec,
    }),
    []
  );

  const integrationDeps = useMemo<AgenticResearchDeps>(() => ({ state, actions, api }), [
    state,
    actions,
    api,
  ]);

  const integration = useAgenticResearchIntegration(integrationDeps);
  const activeChartSpec = integration.pcaChartSpec ?? chartSpecs[0] ?? null;

  return useMemo(
    () => ({
      ...integration,
      activeChartSpec,
      chartSpecs,
      formatToolName,
    }),
    [integration, activeChartSpec, chartSpecs]
  );
}
