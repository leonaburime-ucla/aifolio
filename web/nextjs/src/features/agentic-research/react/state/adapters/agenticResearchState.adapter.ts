import {
  useAgenticResearchActions,
  useAgenticResearchState,
} from "@/features/agentic-research/react/state/zustand/agenticResearchStore";
import type { AgenticResearchStatePort } from "@aifolio/contracts/entities/agentic-research";

/**
 * Adapter hook that exposes Agentic Research state/actions through a state port.
 * @returns State and actions consumed by orchestrators.
 */
export function useAgenticResearchStateAdapter(): AgenticResearchStatePort {
  const state = useAgenticResearchState();
  const actions = useAgenticResearchActions();
  return { state, actions };
}
