import { createAgenticResearchApiAdapter } from "@/features/agentic-research/typescript/api/agenticResearchApi.adapter";
import { useAgenticResearchIntegration } from "@/features/agentic-research/typescript/react/hooks/useAgenticResearch.hooks";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/typescript/react/state/adapters/agenticResearchState.adapter";
import type {
  AgenticResearchApiDeps,
  AgenticResearchDeps,
  AgenticResearchOrchestratorModel,
} from "@/features/agentic-research/__types__/typescript/agenticResearch.types";
import { useMemo } from "react";
import type { UseAgenticResearchOrchestratorOptions } from "@/features/agentic-research/__types__/typescript/react/orchestrators/agenticResearchOrchestrator.types";
import {
  formatToolName,
  resolveActiveChartSpec,
} from "@/features/agentic-research/typescript/logic/agenticResearchChart.logic";

/**
 * Orchestrator hook that wires state + API dependencies into the agentic research hooks.
 * @param deps - Optional adapter overrides for orchestrator ports.
 * @returns UI-ready agentic research model for the page.
 */
export function useAgenticResearchOrchestrator(
  options: UseAgenticResearchOrchestratorOptions = {}
): AgenticResearchOrchestratorModel {
  const {
    useStatePort = useAgenticResearchStateAdapter,
    useChartPort = useAgenticResearchChartActionsAdapter,
  } = options;
  const { state, actions } = useStatePort();
  const { chartSpecs, removeChartSpec } = useChartPort();

  const api = useMemo<AgenticResearchApiDeps>(
    () => ({ ...createAgenticResearchApiAdapter({}) }),
    []
  );

  const integrationDeps = useMemo<AgenticResearchDeps>(() => ({ state, actions, api }), [
    state,
    actions,
    api,
  ]);

  const integration = useAgenticResearchIntegration(integrationDeps);
  const activeChartSpec = resolveActiveChartSpec({
    pcaChartSpec: integration.pcaChartSpec,
    chartSpecs,
  });

  return useMemo(
    () => ({
      ...integration,
      activeChartSpec,
      chartSpecs,
      removeChartSpec,
      formatToolName: (name: string) => formatToolName({ name }),
    }),
    [integration, activeChartSpec, chartSpecs, removeChartSpec]
  );
}
