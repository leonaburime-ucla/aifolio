"use client";

import { useCopilotAction } from "@copilotkit/react-core";
import { useAgenticResearchAiSurface } from "@/features/agentic-research/typescript/react/ai/adapters/useAgenticResearchAiSurface";
import {
  AR_ADD_CHART_SPEC_TOOL,
  AR_CLEAR_CHARTS_TOOL,
  AR_REMOVE_CHART_SPEC_TOOL,
  AR_REORDER_CHART_SPECS_TOOL,
  AR_SET_ACTIVE_DATASET_TOOL,
  REMOVE_CHART_SPEC_TOOL_ALIAS,
  REORDER_CHART_SPECS_TOOL_ALIAS,
  SET_ACTIVE_DATASET_TOOL_ALIAS,
} from "@/features/agentic-research/typescript/config/frontendTools.config";
import type { CopilotActionParameter } from "@/features/agentic-research/__types__/typescript/react/ai/agenticResearchAiTools.types";

const ADD_CHART_SPEC_PARAMETERS: CopilotActionParameter[] = [
  { name: "chartSpec", type: "object", required: false, description: "Single chart spec object to render." },
  { name: "chartSpecs", type: "object[]", required: false, description: "Array of chart spec objects to render." },
];

const REMOVE_CHART_SPEC_PARAMETERS: CopilotActionParameter[] = [
  { name: "chart_id", type: "string", required: true, description: "The chart spec id to remove." },
];

const REORDER_CHART_SPECS_PARAMETERS: CopilotActionParameter[] = [
  { name: "ordered_ids", type: "string[]", required: false, description: "Optional complete or partial chart id order. Any missing ids are appended in their current order." },
  { name: "from_index", type: "number", required: false, description: "Optional source index to move from." },
  { name: "to_index", type: "number", required: false, description: "Optional destination index to move to." },
];

const SET_ACTIVE_DATASET_PARAMETERS: CopilotActionParameter[] = [
  { name: "dataset_id", type: "string", required: true, description: "Dataset id from Agentic Research dataset manifest." },
];

/**
 * Purpose: Registers Agentic Research AI tool surface for external callers.
 */
export default function AgenticResearchAiTools() {
  const { handlers } = useAgenticResearchAiSurface();
  const handleAddChartSpecArgs = ({
    chartSpec,
    chartSpecs,
  }: { chartSpec?: unknown; chartSpecs?: unknown[] }) => handlers.addChartSpec({ chartSpec, chartSpecs });

  const handleClearChartsArgs = () => handlers.clearCharts();

  const handleRemoveChartArgs = (args: {
    chart_id?: string;
    chartId?: string;
    id?: string;
  }) => {
    const resolvedId = args.chart_id ?? args.chartId ?? args.id ?? "";
    return handlers.removeChartSpec(resolvedId);
  };

  const handleReorderArgs = (args: {
    ordered_ids?: string[];
    orderedIds?: string[];
    from_index?: number;
    to_index?: number;
    fromIndex?: number;
    toIndex?: number;
  }) =>
    handlers.reorderChartSpecs({
      ordered_ids: args.ordered_ids ?? args.orderedIds,
      from_index: args.from_index ?? args.fromIndex,
      to_index: args.to_index ?? args.toIndex,
    });

  const handleSetDatasetArgs = (args: {
    dataset_id?: string;
    datasetId?: string;
    dataset?: string;
    id?: string;
  }) => {
    const resolvedDataset = args.dataset_id ?? args.datasetId ?? args.dataset ?? args.id ?? "";
    return handlers.setActiveDataset(resolvedDataset);
  };

  useCopilotAction(
    {
      name: AR_ADD_CHART_SPEC_TOOL,
      description: "Agentic Research only: add one chart spec or an array of chart specs to the local Agentic Research chart store.",
      parameters: ADD_CHART_SPEC_PARAMETERS,
      handler: handleAddChartSpecArgs,
    },
    [handleAddChartSpecArgs]
  );

  useCopilotAction(
    {
      name: AR_CLEAR_CHARTS_TOOL,
      description: "Agentic Research only: clear all currently rendered charts.",
      parameters: [],
      handler: handleClearChartsArgs,
    },
    [handleClearChartsArgs]
  );

  useCopilotAction(
    {
      name: AR_REMOVE_CHART_SPEC_TOOL,
      description: "Agentic Research only: remove one chart by chart_id.",
      parameters: REMOVE_CHART_SPEC_PARAMETERS,
      handler: handleRemoveChartArgs,
    },
    [handleRemoveChartArgs]
  );

  useCopilotAction(
    {
      name: REMOVE_CHART_SPEC_TOOL_ALIAS,
      description: "Alias for Agentic Research remove chart tool.",
      parameters: REMOVE_CHART_SPEC_PARAMETERS,
      handler: handleRemoveChartArgs,
    },
    [handleRemoveChartArgs]
  );

  useCopilotAction(
    {
      name: AR_REORDER_CHART_SPECS_TOOL,
      description: "Agentic Research only: reorder charts either by ordered_ids array or by moving one chart from from_index to to_index.",
      parameters: REORDER_CHART_SPECS_PARAMETERS,
      handler: handleReorderArgs,
    },
    [handleReorderArgs]
  );

  useCopilotAction(
    {
      name: REORDER_CHART_SPECS_TOOL_ALIAS,
      description: "Alias for Agentic Research reorder charts tool.",
      parameters: REORDER_CHART_SPECS_PARAMETERS,
      handler: handleReorderArgs,
    },
    [handleReorderArgs]
  );

  useCopilotAction(
    {
      name: AR_SET_ACTIVE_DATASET_TOOL,
      description: "Agentic Research only: set the active dataset in the dataset combobox by dataset_id.",
      parameters: SET_ACTIVE_DATASET_PARAMETERS,
      handler: handleSetDatasetArgs,
    },
    [handleSetDatasetArgs]
  );

  useCopilotAction(
    {
      name: SET_ACTIVE_DATASET_TOOL_ALIAS,
      description: "Alias for Agentic Research set active dataset tool.",
      parameters: SET_ACTIVE_DATASET_PARAMETERS,
      handler: handleSetDatasetArgs,
    },
    [handleSetDatasetArgs]
  );

  return null;
}
