"use client";

import { useCallback, useMemo, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useCopilotAction } from "@copilotkit/react-core";
import { useCopilotChartActionsAdapter } from "@/features/recharts/state/adapters/chartActions.adapter";
import {
  handleAddChartSpec,
  handleNavigateToPage,
  handleTrainPytorchModel,
} from "@/features/copilot-chat/orchestrators/frontendTools.orchestrator";
import {
  ADD_CHART_SPEC_TOOL,
  CLEAR_CHARTS_TOOL,
  NAVIGATE_TO_PAGE_TOOL,
  TRAIN_PYTORCH_MODEL_TOOL,
} from "@/features/copilot-chat/config/frontendTools.config";
import type { PytorchTrainRequest } from "@/features/ml/api/pytorchApi";

type ClearChartsResponse = {
  status: "ok";
  cleared: true;
};

type NavigateResult =
  | { status: "ok"; resolvedRoute: string }
  | { status: "error"; code: "INVALID_ROUTE"; allowedRoutes: string[] };

/**
 * Return type for the Copilot frontend tools orchestrator hook.
 */
export type CopilotFrontendToolsOrchestrator = {
  handlers: {
    addChartSpec: (payload: {
      chartSpec?: unknown;
      chartSpecs?: unknown[];
    }) => ReturnType<typeof handleAddChartSpec>;
    clearCharts: () => ClearChartsResponse;
    navigateToPage: (route: string) => NavigateResult;
    trainPytorchModel: (payload: PytorchTrainRequest) => ReturnType<typeof handleTrainPytorchModel>;
  };
  router: ReturnType<typeof useRouter>;
};

/**
 * Handler for clearing all charts.
 */
export function handleClearCharts(clearFn: () => void): ClearChartsResponse {
  clearFn();
  return { status: "ok", cleared: true };
}

/**
 * Orchestrator hook for Copilot frontend tools.
 *
 * This hook is self-contained and registers all useCopilotAction handlers internally.
 * It should be called from a component/provider that's rendered within the CopilotKit context.
 *
 * Previously, the useCopilotAction calls were in a separate component (CopilotFrontendTools.tsx).
 * Now they are consolidated here, eliminating the need for "invisible" null-returning components.
 */
export function useCopilotFrontendToolsOrchestrator(): CopilotFrontendToolsOrchestrator {
  // Import adapters internally - view should not import these directly
  const chartActions = useCopilotChartActionsAdapter();
  const router = useRouter();

  // Create stable handler references
  const handleAddChart = useCallback(
    (payload: { chartSpec?: unknown; chartSpecs?: unknown[] }) => {
      return handleAddChartSpec(payload, chartActions.addChartSpec);
    },
    [chartActions.addChartSpec]
  );

  const handleClear = useCallback(() => {
    return handleClearCharts(chartActions.clearChartSpecs);
  }, [chartActions.clearChartSpecs]);

  const handleNavigate = useCallback((route: string) => {
    return handleNavigateToPage(route);
  }, []);

  const handleTrain = useCallback((payload: PytorchTrainRequest) => {
    return handleTrainPytorchModel(payload);
  }, []);

  // Memoize handlers object to prevent unnecessary re-renders
  const handlers = useMemo(
    () => ({
      addChartSpec: handleAddChart,
      clearCharts: handleClear,
      navigateToPage: handleNavigate,
      trainPytorchModel: handleTrain,
    }),
    [handleAddChart, handleClear, handleNavigate, handleTrain]
  );

  // Log registration on mount
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

  // Register Copilot actions internally (previously in CopilotFrontendTools component)
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
        const result = handleAddChart({ chartSpec, chartSpecs });
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
    [handleAddChart]
  );

  useCopilotAction(
    {
      name: CLEAR_CHARTS_TOOL,
      description: "Clear all chart specs from the frontend chart store.",
      parameters: [],
      handler: () => {
        const result = handleClear();
        console.log("[copilot-frontend-tool] clear_charts.success");
        return result;
      },
    },
    [handleClear]
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
        const result = handleNavigate(route);
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
    [router, handleNavigate]
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
        task?: string;
        epochs?: number;
        batch_size?: number;
        learning_rate?: number;
      }) => {
        const validTask = task as "classification" | "regression" | "auto" | undefined;
        return handleTrain({
          dataset_id,
          target_column,
          task: validTask,
          epochs,
          batch_size,
          learning_rate,
        });
      },
    },
    [handleTrain]
  );

  return {
    handlers,
    router,
  };
}
