import { useChartStore } from "@/features/recharts/state/zustand/chartStore";
import type { ChatChartActionsPort } from "@/features/ai/types/chat.types";
import type { ChartSpec } from "@/features/ai/types/chart.types";

/**
 * Extended port for chart actions used by Copilot frontend tools.
 */
export type CopilotChartActionsPort = ChatChartActionsPort & {
  clearChartSpecs: () => void;
};

/**
 * Adapter that exposes chart write operations via a narrow port.
 */
export function useRechartsChartActionsAdapter(): ChatChartActionsPort {
  const addChartSpec = useChartStore((state) => state.addChartSpec);
  return { addChartSpec };
}

/**
 * Extended adapter that exposes additional chart operations for Copilot frontend tools.
 * Includes addChartSpec and clearChartSpecs.
 */
export function useCopilotChartActionsAdapter(): CopilotChartActionsPort {
  const addChartSpec = useChartStore((state) => state.addChartSpec);
  const clearChartSpecs = useChartStore((state) => state.clearChartSpecs);
  return { addChartSpec, clearChartSpecs };
}
