import type { ChartManagementPort } from "@/features/recharts/__types__/typescript/react/state/chartManagementAdapter.types";

export type ChartIntegration = ChartManagementPort;
export type UseChartManagementAdapter = () => ChartManagementPort;

export type UseChartOrchestratorParams = {
  orchestrator?: UseChartManagementAdapter;
};
