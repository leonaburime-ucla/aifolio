"use client";

import { useCopilotAction } from "@copilotkit/react-core";
import { useAgenticResearchChartStore } from "@/features/agentic-research/state/zustand/agenticResearchChartStore";
import {
  useAgenticResearchActions,
  useAgenticResearchState,
} from "@/features/agentic-research/state/zustand/agenticResearchStore";
import { handleAddChartSpec } from "@/features/copilot-chat/orchestrators/frontendTools.orchestrator";

const AR_ADD_CHART_SPEC_TOOL = "ar-add_chart_spec";
const AR_CLEAR_CHARTS_TOOL = "ar-clear_charts";
const AR_REMOVE_CHART_SPEC_TOOL = "ar-remove_chart_spec";
const AR_REORDER_CHART_SPECS_TOOL = "ar-reorder_chart_specs";
const AR_SET_ACTIVE_DATASET_TOOL = "ar-set_active_dataset";
const ADD_CHART_SPEC_TOOL_ALIAS = "add_chart_spec";
const CLEAR_CHARTS_TOOL_ALIAS = "clear_charts";

export default function AgenticResearchFrontendTools() {
  const addChartSpec = useAgenticResearchChartStore((state) => state.addChartSpec);
  const clearChartSpecs = useAgenticResearchChartStore((state) => state.clearChartSpecs);
  const removeChartSpec = useAgenticResearchChartStore((state) => state.removeChartSpec);
  const reorderChartSpecs = useAgenticResearchChartStore((state) => state.reorderChartSpecs);
  const { datasetManifest } = useAgenticResearchState();
  const { setSelectedDatasetId } = useAgenticResearchActions();

  useCopilotAction(
    {
      name: AR_ADD_CHART_SPEC_TOOL,
      description:
        "Agentic Research only: add one chart spec or an array of chart specs to the local Agentic Research chart store.",
      parameters: [
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
      ],
      handler: ({ chartSpec, chartSpecs }: { chartSpec?: unknown; chartSpecs?: unknown[] }) => {
        return handleAddChartSpec({ chartSpec, chartSpecs }, addChartSpec);
      },
    },
    [addChartSpec]
  );

  useCopilotAction(
    {
      name: ADD_CHART_SPEC_TOOL_ALIAS,
      description:
        "Alias for ar-add_chart_spec on /agentic-research. Add one chart spec or an array of chart specs to the local Agentic Research chart store.",
      parameters: [
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
      ],
      handler: ({ chartSpec, chartSpecs }: { chartSpec?: unknown; chartSpecs?: unknown[] }) => {
        return handleAddChartSpec({ chartSpec, chartSpecs }, addChartSpec);
      },
    },
    [addChartSpec]
  );

  useCopilotAction(
    {
      name: AR_CLEAR_CHARTS_TOOL,
      description: "Agentic Research only: clear all currently rendered charts.",
      parameters: [],
      handler: () => {
        clearChartSpecs();
        return { status: "ok", cleared: true };
      },
    },
    [clearChartSpecs]
  );

  useCopilotAction(
    {
      name: CLEAR_CHARTS_TOOL_ALIAS,
      description: "Alias for ar-clear_charts on /agentic-research. Clear all charts.",
      parameters: [],
      handler: () => {
        clearChartSpecs();
        return { status: "ok", cleared: true };
      },
    },
    [clearChartSpecs]
  );

  useCopilotAction(
    {
      name: AR_REMOVE_CHART_SPEC_TOOL,
      description: "Agentic Research only: remove one chart by chart_id.",
      parameters: [
        {
          name: "chart_id",
          type: "string",
          required: true,
          description: "The chart spec id to remove.",
        },
      ],
      handler: ({ chart_id }: { chart_id: string }) => {
        const current = useAgenticResearchChartStore.getState().chartSpecs;
        const exists = current.some((spec) => spec.id === chart_id);
        if (!exists) {
          return {
            status: "error",
            code: "CHART_NOT_FOUND",
            chart_id,
            available_chart_ids: current.map((spec) => spec.id),
          };
        }
        removeChartSpec(chart_id);
        return {
          status: "ok",
          removed_chart_id: chart_id,
          remaining_count: useAgenticResearchChartStore.getState().chartSpecs.length,
        };
      },
    },
    [removeChartSpec]
  );

  useCopilotAction(
    {
      name: AR_REORDER_CHART_SPECS_TOOL,
      description:
        "Agentic Research only: reorder charts either by ordered_ids array or by moving one chart from from_index to to_index.",
      parameters: [
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
      ],
      handler: ({
        ordered_ids,
        from_index,
        to_index,
      }: {
        ordered_ids?: string[];
        from_index?: number;
        to_index?: number;
      }) => {
        const current = useAgenticResearchChartStore.getState().chartSpecs;
        if (Array.isArray(ordered_ids) && ordered_ids.length > 0) {
          reorderChartSpecs(ordered_ids);
          return {
            status: "ok",
            mode: "ordered_ids",
            chart_ids: useAgenticResearchChartStore.getState().chartSpecs.map((spec) => spec.id),
          };
        }

        if (
          typeof from_index === "number" &&
          typeof to_index === "number" &&
          Number.isInteger(from_index) &&
          Number.isInteger(to_index)
        ) {
          if (
            from_index < 0 ||
            from_index >= current.length ||
            to_index < 0 ||
            to_index >= current.length
          ) {
            return {
              status: "error",
              code: "INDEX_OUT_OF_RANGE",
              from_index,
              to_index,
              chart_count: current.length,
            };
          }
          const ids = current.map((spec) => spec.id);
          const [moved] = ids.splice(from_index, 1);
          ids.splice(to_index, 0, moved);
          reorderChartSpecs(ids);
          return {
            status: "ok",
            mode: "index_move",
            chart_ids: useAgenticResearchChartStore.getState().chartSpecs.map((spec) => spec.id),
          };
        }

        return {
          status: "error",
          code: "INVALID_REORDER_PAYLOAD",
          hint: "Provide ordered_ids or both from_index and to_index.",
        };
      },
    },
    [reorderChartSpecs]
  );

  useCopilotAction(
    {
      name: AR_SET_ACTIVE_DATASET_TOOL,
      description:
        "Agentic Research only: set the active dataset in the dataset combobox by dataset_id.",
      parameters: [
        {
          name: "dataset_id",
          type: "string",
          required: true,
          description: "Dataset id from Agentic Research dataset manifest.",
        },
      ],
      handler: ({ dataset_id }: { dataset_id: string }) => {
        const allowedIds = datasetManifest.map((entry) => entry.id);
        if (!allowedIds.includes(dataset_id)) {
          return {
            status: "error",
            code: "INVALID_DATASET_ID",
            dataset_id,
            allowed_dataset_ids: allowedIds,
          };
        }
        useAgenticResearchChartStore.getState().clearChartSpecs();
        setSelectedDatasetId(dataset_id);
        return {
          status: "ok",
          active_dataset_id: dataset_id,
        };
      },
    },
    [datasetManifest, setSelectedDatasetId]
  );

  return null;
}
