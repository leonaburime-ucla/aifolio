import type {
  UseAgenticResearchChartActionsPort,
  UseAgenticResearchStatePort,
} from "@aifolio/contracts/entities/agentic-research";

export type UseAgenticResearchOrchestratorOptions = {
  useStatePort?: UseAgenticResearchStatePort;
  useChartPort?: UseAgenticResearchChartActionsPort;
};
