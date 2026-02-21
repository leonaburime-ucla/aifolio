import {
  useAgenticResearchActions,
  useAgenticResearchState,
} from "@/features/agentic-research/state/zustand/agenticResearchStore";
import type { AgenticResearchStatePort } from "@/features/agentic-research/types/agenticResearch.types";

/**
 * Adapter hook that exposes Agentic Research state/actions through a state port.
 * @returns State and actions consumed by orchestrators.
 */
export function useAgenticResearchStateAdapter(): AgenticResearchStatePort {
  const state = useAgenticResearchState();
  const actions = useAgenticResearchActions();
  return { state, actions };
}
