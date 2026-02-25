"use client";

import { useCopilotAction } from "@copilotkit/react-core";
import { useAgenticResearchFrontendToolsOrchestrator } from "@/features/agentic-research/orchestrators/agenticResearchFrontendTools.orchestrator";
import {
  AR_ADD_CHART_SPEC_TOOL,
  AR_CLEAR_CHARTS_TOOL,
  AR_REMOVE_CHART_SPEC_TOOL,
  AR_REORDER_CHART_SPECS_TOOL,
  AR_SET_ACTIVE_DATASET_TOOL,
  ADD_CHART_SPEC_TOOL_ALIAS,
  CLEAR_CHARTS_TOOL_ALIAS,
} from "@/features/agentic-research/config/frontendTools.config";

const ADD_CHART_SPEC_PARAMETERS: any = [
  {
    name: "chartSpec",
    type: "object",
    required: false,
    description: "Single chart spec object to render.",
  },
  {
    name: "chartSpecs",
    type: "object[]",
    required: false,
    description: "Array of chart spec objects to render.",
  },
];

const REMOVE_CHART_SPEC_PARAMETERS: any = [
  {
    name: "chart_id",
    type: "string",
    required: true,
    description: "The chart spec id to remove.",
  },
];

const REORDER_CHART_SPECS_PARAMETERS: any = [
  {
    name: "ordered_ids",
    type: "string[]",
    required: false,
    description:
      "Optional complete or partial chart id order. Any missing ids are appended in their current order.",
  },
  {
    name: "from_index",
    type: "number",
    required: false,
    description: "Optional source index to move from.",
  },
  {
    name: "to_index",
    type: "number",
    required: false,
    description: "Optional destination index to move to.",
  },
];

const SET_ACTIVE_DATASET_PARAMETERS: any = [
  {
    name: "dataset_id",
    type: "string",
    required: true,
    description: "Dataset id from Agentic Research dataset manifest.",
  },
];

export default function AgenticResearchFrontendTools() {
  const { handlers, datasetManifest } = useAgenticResearchFrontendToolsOrchestrator();

  useCopilotAction(
    {
      name: AR_ADD_CHART_SPEC_TOOL,
      description:
        "Agentic Research only: add one chart spec or an array of chart specs to the local Agentic Research chart store.",
      parameters: ADD_CHART_SPEC_PARAMETERS,
      handler: ({ chartSpec, chartSpecs }: any) => {
        return handlers.addChartSpec({ chartSpec, chartSpecs });
      },
    },
    [handlers.addChartSpec]
  );

  useCopilotAction(
    {
      name: ADD_CHART_SPEC_TOOL_ALIAS,
      description:
        "Alias for ar-add_chart_spec on /agentic-research. Add one chart spec or an array of chart specs to the local Agentic Research chart store.",
      parameters: ADD_CHART_SPEC_PARAMETERS,
      handler: ({ chartSpec, chartSpecs }: any) => {
        return handlers.addChartSpec({ chartSpec, chartSpecs });
      },
    },
    [handlers.addChartSpec]
  );

  useCopilotAction(
    {
      name: AR_CLEAR_CHARTS_TOOL,
      description: "Agentic Research only: clear all currently rendered charts.",
      parameters: [],
      handler: () => handlers.clearCharts(),
    },
    [handlers.clearCharts]
  );

  useCopilotAction(
    {
      name: CLEAR_CHARTS_TOOL_ALIAS,
      description: "Alias for ar-clear_charts on /agentic-research. Clear all charts.",
      parameters: [],
      handler: () => handlers.clearCharts(),
    },
    [handlers.clearCharts]
  );

  useCopilotAction(
    {
      name: AR_REMOVE_CHART_SPEC_TOOL,
      description: "Agentic Research only: remove one chart by chart_id.",
      parameters: REMOVE_CHART_SPEC_PARAMETERS,
      handler: ({ chart_id }: any) => {
        return handlers.removeChartSpec(chart_id);
      },
    },
    [handlers.removeChartSpec]
  );

  useCopilotAction(
    {
      name: AR_REORDER_CHART_SPECS_TOOL,
      description:
        "Agentic Research only: reorder charts either by ordered_ids array or by moving one chart from from_index to to_index.",
      parameters: REORDER_CHART_SPECS_PARAMETERS,
      handler: ({ ordered_ids, from_index, to_index }: any) => {
        return handlers.reorderChartSpecs({ ordered_ids, from_index, to_index });
      },
    },
    [handlers.reorderChartSpecs]
  );

  useCopilotAction(
    {
      name: AR_SET_ACTIVE_DATASET_TOOL,
      description:
        "Agentic Research only: set the active dataset in the dataset combobox by dataset_id.",
      parameters: SET_ACTIVE_DATASET_PARAMETERS,
      handler: ({ dataset_id }: any) => {
        return handlers.setActiveDataset(dataset_id);
      },
    },
    [datasetManifest, handlers.setActiveDataset]
  );

  return null;
}
