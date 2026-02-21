"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useCopilotAction } from "@copilotkit/react-core";
import { useChartStore } from "@/features/recharts/state/zustand/chartStore";
import {
  ADD_CHART_SPEC_TOOL,
  CLEAR_CHARTS_TOOL,
  NAVIGATE_TO_PAGE_TOOL,
  TRAIN_PYTORCH_MODEL_TOOL,
} from "@/features/copilot-chat/config/frontendTools.config";
import {
  handleAddChartSpec,
  handleNavigateToPage,
  handleTrainPytorchModel,
} from "@/features/copilot-chat/orchestrators/frontendTools.orchestrator";

/**
 * Registers browser-side Copilot tools.
 *
 * This component has no UI; mounting it enables frontend tool execution for
 * the surrounding Copilot session.
 */
export default function CopilotFrontendTools() {
  const addChartSpec = useChartStore((state) => state.addChartSpec);
  const clearChartSpecs = useChartStore((state) => state.clearChartSpecs);
  const router = useRouter();

  useEffect(() => {
    console.log("[copilot-frontend-tool] registered", {
      names: [
        ADD_CHART_SPEC_TOOL,
        CLEAR_CHARTS_TOOL,
        NAVIGATE_TO_PAGE_TOOL,
        TRAIN_PYTORCH_MODEL_TOOL,
      ],
    });
  }, []);

  useCopilotAction(
    {
      name: ADD_CHART_SPEC_TOOL,
      description:
        "Add one chart spec or an array of chart specs to the frontend chart store for immediate rendering.",
      parameters: [
        {
          name: "chartSpec",
          type: "object",
          required: false,
          description: "A single chart spec object to normalize and render.",
        },
        {
          name: "chartSpecs",
          type: "object[]",
          required: false,
          description: "An array of chart spec objects to normalize and render.",
        },
      ],
      handler: ({ chartSpec, chartSpecs }: { chartSpec?: unknown; chartSpecs?: unknown[] }) => {
        const result = handleAddChartSpec({ chartSpec, chartSpecs }, addChartSpec);
        if (result.status === "error") {
          console.warn("[copilot-frontend-tool] add_chart_spec.invalid_payload", {
            chartSpec,
            chartSpecs,
          });
          return result;
        }

        console.log("[copilot-frontend-tool] add_chart_spec.success", {
          addedCount: result.addedCount,
          ids: result.ids,
        });

        return result;
      },
    },
    [addChartSpec],
  );

  useCopilotAction(
    {
      name: CLEAR_CHARTS_TOOL,
      description: "Clear all chart specs from the frontend chart store.",
      parameters: [],
      handler: () => {
        clearChartSpecs();
        console.log("[copilot-frontend-tool] clear_charts.success");
        return {
          status: "ok",
          cleared: true,
        };
      },
    },
    [clearChartSpecs],
  );

  useCopilotAction(
    {
      name: NAVIGATE_TO_PAGE_TOOL,
      description:
        "Navigate the user to another app page. Allowed targets: /, /ag-ui, /agentic-research, /ml/pytorch, /ml/tensorflow, /ml/knowledge-distillation.",
      parameters: [
        {
          name: "route",
          type: "string",
          required: true,
          description: "Route path or alias, such as '/', 'ag-ui', 'agentic research', or '/ml/pytorch'.",
        },
      ],
      handler: ({ route }: { route: string }) => {
        const result = handleNavigateToPage(route);
        if (result.status === "error") {
          console.warn("[copilot-frontend-tool] navigate_to_page.invalid_route", {
            requestedRoute: route,
          });
          return result;
        }

        router.push(result.resolvedRoute);
        console.log("[copilot-frontend-tool] navigate_to_page.success", {
          requestedRoute: route,
          resolvedRoute: result.resolvedRoute,
        });
        return result;
      },
    },
    [router],
  );

  useCopilotAction(
    {
      name: TRAIN_PYTORCH_MODEL_TOOL,
      description:
        "Start one backend PyTorch training run using a dataset from /ml-data. Returns model_id, model_path, and metrics.",
      parameters: [
        {
          name: "dataset_id",
          type: "string",
          required: true,
          description:
            "Dataset file id from /ml-data (for example customer_churn_telco.csv).",
        },
        {
          name: "target_column",
          type: "string",
          required: true,
          description: "Target column name in the selected dataset.",
        },
        {
          name: "task",
          type: "string",
          required: false,
          description:
            "Optional task hint: classification, regression, or auto.",
        },
        {
          name: "epochs",
          type: "number",
          required: false,
          description: "Optional training epochs (default 200).",
        },
        {
          name: "batch_size",
          type: "number",
          required: false,
          description: "Optional batch size (default 64).",
        },
        {
          name: "learning_rate",
          type: "number",
          required: false,
          description: "Optional learning rate (default 0.001).",
        },
      ],
      handler: async ({
        dataset_id,
        target_column,
        task,
        epochs,
        batch_size,
        learning_rate,
      }: {
        dataset_id: string;
        target_column: string;
        task?: "classification" | "regression" | "auto";
        epochs?: number;
        batch_size?: number;
        learning_rate?: number;
      }) => {
        return handleTrainPytorchModel({
          dataset_id,
          target_column,
          task,
          epochs,
          batch_size,
          learning_rate,
        });
      },
    },
    [],
  );

  return null;
}
