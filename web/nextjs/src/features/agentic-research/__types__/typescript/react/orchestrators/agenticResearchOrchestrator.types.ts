import type {
  UseAgenticResearchChartActionsPort,
  UseAgenticResearchStatePort,
} from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

export type UseAgenticResearchOrchestratorOptions = {
  useStatePort?: UseAgenticResearchStatePort;
  useChartPort?: UseAgenticResearchChartActionsPort;
};
