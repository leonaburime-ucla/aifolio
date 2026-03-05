import type { AgenticResearchActions } from "@/features/agentic-research/__types__/typescript/agenticResearch.types";

export type ApplyDatasetLoadResetInput = {
  actions: Pick<
    AgenticResearchActions,
    | "setTableRows"
    | "setTableColumns"
    | "setNumericMatrix"
    | "setFeatureNames"
    | "setPcaChartSpec"
  >;
};
