import { normalizeChartSpecInput } from "@/features/copilot-chat/utils/copilotAssistantPayload.util";
import type { ChartSpec } from "@/features/ai/types/chart.types";
import {
  isAllowedRoute,
  resolveRouteAlias,
  ROUTE_ALIASES,
} from "@/features/copilot-chat/config/frontendTools.config";
import {
  trainPytorchModel,
  type PytorchTrainRequest,
} from "@/features/ml/api/pytorchApi";

export function handleAddChartSpec(
  payload: { chartSpec?: unknown; chartSpecs?: unknown[] },
  addChartSpec: (spec: ChartSpec) => void
) {
  const combinedPayload = payload.chartSpecs ?? payload.chartSpec;
  const normalized = normalizeChartSpecInput(combinedPayload);

  if (!normalized) {
    return {
      status: "error",
      code: "INVALID_CHART_SPEC",
      addedCount: 0,
    };
  }

  const specs = Array.isArray(normalized) ? normalized : [normalized];
  specs.forEach((spec) => addChartSpec(spec));

  return {
    status: "ok",
    addedCount: specs.length,
    ids: specs.map((spec) => spec.id),
  };
}

export function handleNavigateToPage(route: string):
  | { status: "ok"; resolvedRoute: string }
  | { status: "error"; code: "INVALID_ROUTE"; allowedRoutes: string[] } {
  const resolvedRoute = resolveRouteAlias(route);
  if (!resolvedRoute || !isAllowedRoute(resolvedRoute)) {
    return {
      status: "error",
      code: "INVALID_ROUTE",
      allowedRoutes: Array.from(new Set(Object.values(ROUTE_ALIASES))),
    };
  }

  return {
    status: "ok",
    resolvedRoute,
  };
}

export async function handleTrainPytorchModel(payload: PytorchTrainRequest) {
  return trainPytorchModel(payload);
}
