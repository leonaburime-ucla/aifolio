import { useChartManagementAdapter } from "@/features/recharts/react/state/adapters/chartManagement.adapter";
import type {
  ChartIntegration,
  UseChartOrchestratorParams,
} from "@/features/recharts/react/orchestrators/chartOrchestrator.types";

export type { ChartIntegration } from "@/features/recharts/react/orchestrators/chartOrchestrator.types";

/**
 * Chart orchestrator that exposes chart state/actions through an injectable port.
 */
export function useChartOrchestrator({
  orchestrator = useChartManagementAdapter,
}: UseChartOrchestratorParams = {}): ChartIntegration {
  return orchestrator();
}
