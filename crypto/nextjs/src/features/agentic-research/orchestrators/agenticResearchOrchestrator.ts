import { fetchPcaChartSpec } from "@/features/agentic-research/api/agenticResearchApi";
import { useAgenticResearchIntegration } from "@/features/agentic-research/hooks/useAgenticResearch.hooks";
import {
  useAgenticResearchActions,
  useAgenticResearchState,
} from "@/features/agentic-research/state/zustand/agenticResearchStore";
import type {
  AgenticResearchActions,
  AgenticResearchApiDeps,
  AgenticResearchDeps,
  AgenticResearchIntegration,
  AgenticResearchState,
} from "@/features/agentic-research/types/agenticResearch.types";
import { useMemo } from "react";

/**
 * Orchestrator hook that wires state + API dependencies into the agentic research hooks.
 */
export function useAgenticResearchOrchestrator(): AgenticResearchIntegration {
  const state = useAgenticResearchState();
  const actions = useMemo(() => useAgenticResearchActions(), []);

  const api = useMemo<AgenticResearchApiDeps>(
    () => ({
      fetchPcaChartSpec,
    }),
    []
  );

  const deps = useMemo<AgenticResearchDeps>(() => ({ state, actions, api }), [
    state,
    actions,
    api,
  ]);

  return useAgenticResearchIntegration(deps);
}
