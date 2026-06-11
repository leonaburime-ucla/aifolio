import { createAgenticResearchApiAdapter } from "@/features/agentic-research/api/agenticResearchApi.adapter";
import { useAgenticResearchIntegration } from "@/features/agentic-research/react/hooks/useAgenticResearch.hooks";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/react/state/adapters/chartActions.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/react/state/adapters/agenticResearchState.adapter";
import type {
  AgenticResearchApiDeps,
  AgenticResearchDeps,
  AgenticResearchOrchestratorModel,
} from "@aifolio/contracts/entities/agentic-research";
import { useMemo } from "react";
import type { UseAgenticResearchOrchestratorOptions } from "@/features/agentic-research/react/orchestrators/agenticResearchOrchestrator.types";
import {
  formatToolName,
  resolveActiveChartSpec,
} from "@aifolio/frontend-core/agentic-research";

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
