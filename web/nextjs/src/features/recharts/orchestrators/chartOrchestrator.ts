import { useChartManagementAdapter } from "@/features/recharts/state/adapters/chartManagement.adapter";

export type ChartIntegration = ReturnType<typeof useChartManagementAdapter>;

/**
 * Chart orchestrator that exposes chart state/actions through an injectable port.
 */
export function useChartOrchestrator({
  useChartManagementPort = useChartManagementAdapter,
}: {
  useChartManagementPort?: typeof useChartManagementAdapter;
} = {}): ChartIntegration {
  return useChartManagementPort();
}
