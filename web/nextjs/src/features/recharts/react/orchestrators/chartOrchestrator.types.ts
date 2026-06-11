import type { ChartManagementPort } from "@aifolio/contracts/entities/chart";

export type ChartIntegration = ChartManagementPort;
export type UseChartManagementAdapter = () => ChartManagementPort;

export type UseChartOrchestratorParams = {
  orchestrator?: UseChartManagementAdapter;
};
