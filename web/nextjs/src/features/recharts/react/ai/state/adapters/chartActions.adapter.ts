import type { CopilotChartActionsPort } from "@aifolio/contracts/entities/chart";
import { useAiChartStore } from "@/features/recharts/react/ai/state/zustand/aiChartStore";

/**
 * AI tooling adapter exposing chart write operations for Copilot/external agents.
 */
export function useCopilotChartActionsAdapter(): CopilotChartActionsPort {
  const addChartSpec = useAiChartStore((state) => state.addChartSpec);
  const clearChartSpecs = useAiChartStore((state) => state.clearChartSpecs);
  return { addChartSpec, clearChartSpecs };
}
