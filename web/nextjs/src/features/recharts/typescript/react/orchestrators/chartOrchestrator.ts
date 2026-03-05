import { useChartManagementAdapter } from "@/features/recharts/typescript/react/state/adapters/chartManagement.adapter";
import type {
  ChartIntegration,
  UseChartOrchestratorParams,
} from "@/features/recharts/__types__/typescript/react/orchestrators/chartOrchestrator.types";

export type { ChartIntegration } from "@/features/recharts/__types__/typescript/react/orchestrators/chartOrchestrator.types";

/**
 * Chart orchestrator that exposes chart state/actions through an injectable port.
 */
export function useChartOrchestrator({
  orchestrator = useChartManagementAdapter,
}: UseChartOrchestratorParams = {}): ChartIntegration {
  return orchestrator();
}
